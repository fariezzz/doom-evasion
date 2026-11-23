"""Run a local interactive ViZDoom and live CNN robustness demo.

Start from the project directory:
    py live_demo.py --model runs/baseline/best_model.pt --map defend_the_center

The CNN was trained only on defend_the_center frames, so predictions are
only meaningful on that map. Other maps are playable from the dashboard,
but their predictions are shown with a warning.
"""

import argparse
import base64
import json
import math
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from scripts.train_baseline import SmallCNN


ROOT = Path(__file__).resolve().parent
ALLOWED_EPSILONS = (0.01, 0.03, 0.05)
DEFAULT_MAP = "defend_the_center"

# Map id -> scenario config, display label, and whether the CNN was trained
# on this map. dead_simple is left out: its config path fails to load on
# Windows. Every map here was verified to load with the buttons probed on
# 2026-11-16; re-probe before adding a new one.
MAPS = {
    "defend_the_center": {
        "config": "defend_the_center.cfg",
        "label": "Defend the Center",
        "trained": True,
    },
    "basic": {"config": "basic.cfg", "label": "Basic", "trained": False},
    "deadly_corridor": {
        "config": "deadly_corridor.cfg",
        "label": "Deadly Corridor",
        "trained": False,
    },
    "defend_the_line": {
        "config": "defend_the_line.cfg",
        "label": "Defend the Line",
        "trained": False,
    },
    "health_gathering": {
        "config": "health_gathering.cfg",
        "label": "Health Gathering",
        "trained": False,
    },
    "take_cover": {
        "config": "take_cover.cfg",
        "label": "Take Cover",
        "trained": False,
    },
    "predict_position": {
        "config": "predict_position.cfg",
        "label": "Predict Position",
        "trained": False,
    },
    "my_way_home": {
        "config": "my_way_home.cfg",
        "label": "My Way Home",
        "trained": False,
    },
}

# Mouse look: how many mouse pixels equal one game turn tick, how many
# unspent turn ticks may pile up, and how long queued turning may survive
# without fresh mouse input. ViZDoom turning is one binary button press
# per game tick, so a long queue cannot turn faster, it only keeps the
# camera moving after the mouse stops: the old limit of 45 ticks meant
# up to 1.5 seconds of drift at 30 FPS. Keep the queue short and drop
# stale remainder so the camera stops soon after the mouse does.
TURN_PIXELS_PER_TICK = 5.0
TURN_IMPULSE_LIMIT = 10.0
TURN_IMPULSE_MAX_AGE_SECONDS = 0.12


def to_gray_frame(buffer):
    if buffer.ndim == 3:
        if buffer.shape[0] == 1:
            return buffer[0]
        if buffer.shape[0] == 3:
            return cv2.cvtColor(np.transpose(buffer, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        return buffer[0]
    return buffer


def true_label_from_state(state):
    """Ground-truth label from the game itself, same rule as training: 1 if
    any visible label has the Monster category, else 0."""
    labels = state.labels if state.labels is not None else []
    for label in labels:
        if getattr(label, "object_category", "") == "Monster":
            return 1
    return 0


def encode_image(image):
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise RuntimeError("Could not encode an image for the browser")
    return base64.b64encode(encoded).decode("ascii")


def encode_gray_png(image):
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise RuntimeError("Could not encode an evaluation image")
    return base64.b64encode(encoded).decode("ascii")


class LiveDemo:
    def __init__(self, model_path, fps, map_id):
        if map_id not in MAPS:
            raise ValueError("Unknown map: " + map_id)
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.model = SmallCNN().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "state_dict" not in checkpoint:
            raise ValueError("Model checkpoint does not contain a state_dict: " + model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model_lock = threading.Lock()
        self.lock = threading.Lock()
        self.target_fps = fps
        self.map_id = map_id
        self.maps_info = [
            {"id": map_key, "label": info["label"], "trained": info["trained"]}
            for map_key, info in MAPS.items()
        ]
        self.controls = {
            "left": False,
            "right": False,
            "forward": False,
            "backward": False,
            "attack": False,
        }
        self.turn_impulse = 0.0
        self.turn_updated_at = time.monotonic()
        self.turn_seq = 0
        self.turn_clear_seq = 0
        self.map_requested = None
        self.reset_requested = False
        self.stop_requested = False
        self.state = {
            "ready": False,
            "error": None,
            "frame": None,
            "prediction": None,
            "confidence": None,
            "true_label": None,
            "fps": 0.0,
            "map_id": map_id,
            "map_label": MAPS[map_id]["label"],
            "map_trained": MAPS[map_id]["trained"],
            "available_controls": {name: False for name in self.controls},
            "can_turn": False,
        }
        self.baseline_capture = None
        self.capture_sequence = 0
        self.current_model_input = None
        self.current_true_label = None
        self.baseline_metrics = self._read_json("runs/baseline/test_metrics.json")
        self.attack_metrics = self._read_json("runs/fgsm/attack_results.json")
        self.mitigation_metrics = self._read_json("runs/mitigation/test_metrics.json")
        self.mitigated_attack_metrics = self._read_json("runs/fgsm_mitigated/attack_results.json")

        # The game object is created lazily by the worker thread, so all
        # ViZDoom calls happen on one thread.
        self.game = None
        self.buttons = []
        self.button_indices = {name: None for name in self.controls}
        self.turn_indices = {"left": None, "right": None}
        self.worker = threading.Thread(target=self._run_game, name="vizdoom-loop", daemon=True)
        self.worker.start()

    @staticmethod
    def _read_json(relative_path):
        path = ROOT / relative_path
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _build_game(map_id):
        """Create and start a game for one map. Returns (game, buttons,
        button_indices, turn_indices). Raises on failure without touching
        self, so a failed map switch leaves the running game alone."""
        game = vzd.DoomGame()
        try:
            game.load_config(os.path.join(vzd.scenarios_path, MAPS[map_id]["config"]))
            game.set_window_visible(False)
            game.set_labels_buffer_enabled(True)
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
            game.set_episode_timeout(2100)
            game.init()
        except Exception:
            try:
                game.close()
            except Exception:
                pass
            raise
        buttons = game.get_available_buttons()
        names = [str(button) for button in buttons]

        def find(*keys):
            for key in keys:
                index = next((i for i, name in enumerate(names) if key in name), None)
                if index is not None:
                    return index
            return None

        # Every control is optional: maps offer different buttons.
        # defend_the_center has no MOVE buttons, take_cover has no TURN or
        # ATTACK buttons, so left/right fall back to strafing when a map
        # has no turning buttons.
        button_indices = {
            "left": find("TURN_LEFT", "MOVE_LEFT"),
            "right": find("TURN_RIGHT", "MOVE_RIGHT"),
            "forward": find("MOVE_FORWARD"),
            "backward": find("MOVE_BACKWARD"),
            "attack": find("ATTACK"),
        }
        # Mouse look only turns. It must not fall back to strafing: on a
        # MOVE-only map (basic, take_cover) there is no turning, so mouse
        # movement does nothing instead of sliding the player sideways.
        turn_indices = {
            "left": find("TURN_LEFT"),
            "right": find("TURN_RIGHT"),
        }
        game.new_episode()
        return game, buttons, button_indices, turn_indices

    def _apply_game(self, game, buttons, button_indices, turn_indices, map_id):
        old_game = self.game
        self.game = game
        self.buttons = buttons
        self.button_indices = button_indices
        self.turn_indices = turn_indices
        self.map_id = map_id
        self.turn_impulse = 0.0
        if old_game is not None:
            try:
                old_game.close()
            except Exception:
                pass
        with self.lock:
            self.state["map_id"] = map_id
            self.state["map_label"] = MAPS[map_id]["label"]
            self.state["map_trained"] = MAPS[map_id]["trained"]
            self.state["available_controls"] = {
                name: index is not None for name, index in button_indices.items()
            }
            self.state["can_turn"] = (
                turn_indices["left"] is not None and turn_indices["right"] is not None
            )

    def set_controls(self, controls):
        if not isinstance(controls, dict):
            raise ValueError("Controls must be a JSON object")
        allowed = {"left", "right", "forward", "backward", "attack"}
        if set(controls) - allowed:
            raise ValueError("Unknown control; use forward, backward, left, right, and attack")
        if any(name in controls and not isinstance(value, bool) for name, value in controls.items()):
            raise ValueError("Control values must be true or false")
        with self.lock:
            self.controls = {name: bool(controls.get(name, False)) for name in self.controls}

    def add_turn(self, delta):
        """Add mouse-look movement. Positive delta turns right, negative
        turns left. The game loop spends it one turn tick at a time and
        drops whatever is left once it goes stale, so a fast swipe cannot
        keep the camera turning long after the mouse stops."""
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError("Turn delta must be a number")
        if not math.isfinite(delta) or abs(delta) > 2000:
            raise ValueError("Turn delta is out of range")
        with self.lock:
            self.turn_impulse = max(
                -TURN_IMPULSE_LIMIT,
                min(TURN_IMPULSE_LIMIT, self.turn_impulse + delta / TURN_PIXELS_PER_TICK),
            )
            self.turn_updated_at = time.monotonic()
            self.turn_seq += 1

    def cancel_turn(self):
        """Drop all queued mouse turning at once, for example when the
        browser releases the mouse lock."""
        with self.lock:
            self.turn_impulse = 0.0
            self.turn_clear_seq += 1

    def request_map(self, map_id):
        if not isinstance(map_id, str) or map_id not in MAPS:
            raise ValueError("Unknown map: " + str(map_id))
        with self.lock:
            self.controls = {name: False for name in self.controls}
            self.turn_impulse = 0.0
            self.turn_clear_seq += 1
            self.baseline_capture = None
            self.map_requested = map_id

    def request_reset(self):
        with self.lock:
            self.controls = {name: False for name in self.controls}
            self.turn_impulse = 0.0
            self.turn_clear_seq += 1
            self.reset_requested = True

    def status(self):
        with self.lock:
            current = dict(self.state)
        return {
            **current,
            "maps": self.maps_info,
            "baseline_metrics": self.baseline_metrics,
            "attack_metrics": self.attack_metrics,
            "mitigation_metrics": self.mitigation_metrics,
            "mitigated_attack_metrics": self.mitigated_attack_metrics,
        }

    def capture_current_frame(self):
        with self.lock:
            if (
                not self.state["ready"]
                or self.state["frame"] is None
                or self.current_model_input is None
            ):
                raise RuntimeError(self.state["error"] or "Frame game belum tersedia")
            self.capture_sequence += 1
            self.baseline_capture = {
                "id": self.capture_sequence,
                "frame": self.state["frame"],
                "prediction": self.state["prediction"],
                "confidence": self.state["confidence"],
                "true_label": self.current_true_label,
                "model_input": self.current_model_input.copy(),
                "evaluation": None,
            }
            return self._public_capture(self.baseline_capture)

    def get_baseline_capture(self):
        with self.lock:
            if self.baseline_capture is None:
                return None
            return self._public_capture(self.baseline_capture)

    @staticmethod
    def _public_capture(capture):
        return {
            "frame": capture["frame"],
            "prediction": capture["prediction"],
            "confidence": capture["confidence"],
            "true_label": capture["true_label"],
            "evaluation": capture["evaluation"],
        }

    def evaluate_baseline_capture(self, label):
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Label capture harus 0 (tanpa musuh) atau 1 (ada musuh)")

        with self.lock:
            if self.baseline_capture is None:
                raise RuntimeError("Ambil capture game terlebih dahulu")
            capture_id = self.baseline_capture["id"]
            model_input = self.baseline_capture["model_input"].copy()

        x = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0)
        target = torch.tensor([label], dtype=torch.long, device=self.device)
        with self.model_lock:
            self.model.zero_grad(set_to_none=True)
            x_for_grad = x.clone().detach().requires_grad_(True)
            clean_logits = self.model(x_for_grad)
            F.cross_entropy(clean_logits, target).backward()
            gradient = x_for_grad.grad.detach()
            clean_probabilities = torch.softmax(clean_logits.detach(), dim=1)[0]
            clean_prediction = int(clean_probabilities.argmax().item())
            samples = [{
                "epsilon": 0.0,
                "image": encode_gray_png(model_input),
                "prediction": clean_prediction,
                "confidence": float(clean_probabilities[clean_prediction].item()),
            }]
            for epsilon in ALLOWED_EPSILONS:
                changed = torch.clamp(x + epsilon * torch.sign(gradient), 0.0, 1.0)
                with torch.inference_mode():
                    probabilities = torch.softmax(self.model(changed), dim=1)[0]
                prediction = int(probabilities.argmax().item())
                samples.append({
                    "epsilon": epsilon,
                    "image": encode_gray_png(changed[0, 0].cpu().numpy()),
                    "prediction": prediction,
                    "confidence": float(probabilities[prediction].item()),
                })
            self.model.zero_grad(set_to_none=True)

        result = {
            "label": label,
            "samples": [
                {**sample, "correct": sample["prediction"] == label}
                for sample in samples
            ],
        }
        with self.lock:
            if self.baseline_capture is None or self.baseline_capture["id"] != capture_id:
                raise RuntimeError("Capture berubah saat evaluasi; ambil capture baru lalu ulangi")
            self.baseline_capture["evaluation"] = result
        return result

    def _current_input(self, state):
        gray = to_gray_frame(state.screen_buffer).copy()
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        model_input = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0)
        with self.model_lock, torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        prediction = int(probabilities.argmax().item())
        confidence = float(probabilities[prediction].item())
        return gray, model_input, prediction, confidence, true_label_from_state(state)

    def _run_game(self):
        last_tick = None
        try:
            with self.lock:
                self.state["ready"] = True
            while True:
                tick_started = time.perf_counter()
                elapsed = tick_started - last_tick if last_tick is not None else 0.0
                measured_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                last_tick = tick_started
                with self.lock:
                    if self.stop_requested:
                        break
                    reset = self.reset_requested
                    self.reset_requested = False
                    map_requested = self.map_requested
                    self.map_requested = None
                    controls = dict(self.controls)
                    impulse = self.turn_impulse
                    impulse_seq = self.turn_seq
                    impulse_clear_seq = self.turn_clear_seq
                    self.turn_impulse = 0.0

                if map_requested is not None and map_requested != self.map_id:
                    try:
                        game, buttons, indices, turn_indices = self._build_game(map_requested)
                    except Exception as error:
                        with self.lock:
                            if self.game is None:
                                raise
                            self.state["error"] = "Gagal pindah map: " + str(error)
                        continue
                    self._apply_game(game, buttons, indices, turn_indices, map_requested)
                    impulse = 0.0

                if self.game is None:
                    game, buttons, indices, turn_indices = self._build_game(self.map_id)
                    self._apply_game(game, buttons, indices, turn_indices, self.map_id)

                if reset or self.game.is_episode_finished():
                    self.game.new_episode()

                state = self.game.get_state()
                if state is None:
                    time.sleep(0.01)
                    continue
                gray, model_input, prediction, confidence, true_label = self._current_input(state)
                frame_jpeg = encode_image(gray)
                action = [0] * len(self.buttons)
                for name, pressed in controls.items():
                    button_index = self.button_indices[name]
                    if pressed and button_index is not None:
                        action[button_index] = 1
                if impulse >= 1.0:
                    turn_index = self.turn_indices["right"]
                    if turn_index is not None:
                        action[turn_index] = 1
                        impulse -= 1.0
                elif impulse <= -1.0:
                    turn_index = self.turn_indices["left"]
                    if turn_index is not None:
                        action[turn_index] = 1
                        impulse += 1.0
                with self.lock:
                    # Keep the unspent remainder for the next ticks, plus
                    # input that arrived while this tick was running. Drop
                    # the remainder when the queue was cleared (mouse lock
                    # released, map switch, reset) or when no fresh mouse
                    # input arrived for longer than the maximum age: that
                    # remainder is old movement, and spending it is what
                    # made the camera keep turning after the mouse stopped.
                    cleared = self.turn_clear_seq != impulse_clear_seq
                    stale = (
                        self.turn_seq == impulse_seq
                        and (time.monotonic() - self.turn_updated_at)
                        > TURN_IMPULSE_MAX_AGE_SECONDS
                    )
                    remainder = 0.0 if (cleared or stale) else impulse
                    self.turn_impulse = max(
                        -TURN_IMPULSE_LIMIT,
                        min(TURN_IMPULSE_LIMIT, self.turn_impulse + remainder),
                    )
                self.game.make_action(action, 1)

                with self.lock:
                    self.state.update({
                        "ready": True,
                        "error": None,
                        "frame": frame_jpeg,
                        "prediction": prediction,
                        "confidence": confidence,
                        "true_label": true_label,
                        "fps": measured_fps,
                    })
                    self.current_model_input = model_input.copy()
                    self.current_true_label = true_label
                delay = (1.0 / self.target_fps) - (time.perf_counter() - tick_started)
                if delay > 0:
                    time.sleep(delay)
        except Exception as error:
            with self.lock:
                self.state["error"] = str(error)
                self.state["ready"] = False
        finally:
            if self.game is not None:
                try:
                    self.game.close()
                except Exception:
                    pass

    def close(self):
        with self.lock:
            self.stop_requested = True
        self.worker.join(timeout=3)


class DashboardHandler(SimpleHTTPRequestHandler):
    demo = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format_string, *args):
        return

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length < 1 or length > 4096:
            raise ValueError("Request body must be between 1 and 4096 bytes")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON") from error

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if path == "/api/status":
            return self._send_json(self.demo.status())
        if path == "/api/capture":
            return self._send_json(self.demo.get_baseline_capture() or {})
        if path.startswith("/assets/fgsm/examples_eps_") and path.endswith(".png"):
            name = Path(path).name
            if name in {f"examples_eps_{epsilon:.2f}.png" for epsilon in ALLOWED_EPSILONS}:
                self.path = "/runs/fgsm/" + name
                return super().do_GET()
        if path.startswith("/assets/fgsm-mitigated/examples_eps_") and path.endswith(".png"):
            name = Path(path).name
            if name in {f"examples_eps_{epsilon:.2f}.png" for epsilon in ALLOWED_EPSILONS}:
                self.path = "/runs/fgsm_mitigated/" + name
                return super().do_GET()
        self.send_error(404, "Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send_error(403, "Cross-origin requests are not allowed")
                return
            body = self._read_json()
            if path == "/api/input":
                self.demo.set_controls(body)
                return self._send_json({"ok": True})
            if path == "/api/turn":
                if not isinstance(body, dict) or set(body) - {"delta", "cancel"}:
                    raise ValueError("Kirim delta mouse: {\"delta\": angka}")
                if "cancel" in body and not isinstance(body["cancel"], bool):
                    raise ValueError("Cancel harus true atau false")
                if body.get("cancel") is True:
                    self.demo.cancel_turn()
                    return self._send_json({"ok": True})
                if "delta" not in body:
                    raise ValueError("Kirim delta mouse: {\"delta\": angka}")
                self.demo.add_turn(body["delta"])
                return self._send_json({"ok": True})
            if path == "/api/map":
                if not isinstance(body, dict) or set(body) != {"map"}:
                    raise ValueError("Kirim id map: {\"map\": \"defend_the_center\"}")
                self.demo.request_map(body["map"])
                return self._send_json({"ok": True})
            if path == "/api/reset":
                self.demo.request_reset()
                return self._send_json({"ok": True})
            if path == "/api/capture":
                return self._send_json(self.demo.capture_current_frame())
            if path == "/api/evaluate-capture":
                if not isinstance(body, dict) or set(body) != {"label"}:
                    raise ValueError("Kirim label capture: 0 (tanpa musuh) atau 1 (ada musuh)")
                return self._send_json(self.demo.evaluate_baseline_capture(body["label"]))
            self.send_error(404, "Not found")
        except (ValueError, RuntimeError) as error:
            self._send_json({"error": str(error)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/baseline/best_model.pt")
    parser.add_argument("--map", dest="map_id", default=DEFAULT_MAP, choices=sorted(MAPS))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.fps < 1 or args.fps > 60:
        parser.error("--fps must be between 1 and 60")
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        parser.error("Model checkpoint not found: " + str(model_path))

    demo = LiveDemo(str(model_path), args.fps, args.map_id)
    DashboardHandler.demo = demo
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), DashboardHandler)
    server.daemon_threads = True
    print(f"Dashboard: http://{host}:{args.port}")
    print(f"Map: {MAPS[args.map_id]['label']} (model dilatih pada Defend the Center)")
    print("Controls: A/Left = turn left, D/Right = turn right, W/Up = forward, S/Down = backward, Space = fire")
    print("Mouse: click the game frame, move the mouse to aim, left click to fire.")
    print("Press Ctrl+C to stop the server and game.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ViZDoom and dashboard...")
    finally:
        server.server_close()
        demo.close()


if __name__ == "__main__":
    main()
