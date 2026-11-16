"""Run a local interactive ViZDoom and live CNN robustness demo.

Start from the project directory:
    py live_demo.py --model runs/baseline/best_model.pt
"""

import argparse
import base64
import json
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

from train_baseline import SmallCNN


ROOT = Path(__file__).resolve().parent
ALLOWED_EPSILONS = (0.01, 0.03, 0.05)


def to_gray_frame(buffer):
    if buffer.ndim == 3:
        if buffer.shape[0] == 1:
            return buffer[0]
        if buffer.shape[0] == 3:
            return cv2.cvtColor(np.transpose(buffer, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        return buffer[0]
    return buffer


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
    def __init__(self, model_path, fps):
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
        self.controls = {
            "left": False,
            "right": False,
            "forward": False,
            "backward": False,
            "attack": False,
        }
        self.reset_requested = False
        self.stop_requested = False
        self.state = {
            "ready": False,
            "error": None,
            "frame": None,
            "prediction": None,
            "confidence": None,
            "fps": 0.0,
        }
        self.baseline_capture = None
        self.capture_sequence = 0
        self.current_model_input = None
        self.baseline_metrics = self._read_json("runs/baseline/test_metrics.json")
        self.attack_metrics = self._read_json("runs/fgsm/attack_results.json")

        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
        self.game.set_window_visible(False)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_episode_timeout(2100)
        self.game.init()
        self.buttons = self.game.get_available_buttons()
        button_names = [str(button) for button in self.buttons]
        self.button_indices = {
            "left": next((i for i, name in enumerate(button_names) if "TURN_LEFT" in name), None),
            "right": next((i for i, name in enumerate(button_names) if "TURN_RIGHT" in name), None),
            "forward": next((i for i, name in enumerate(button_names) if "MOVE_FORWARD" in name), None),
            "backward": next((i for i, name in enumerate(button_names) if "MOVE_BACKWARD" in name), None),
            "attack": next((i for i, name in enumerate(button_names) if "ATTACK" in name), None),
        }
        if any(self.button_indices[name] is None for name in ("left", "right", "forward", "backward")):
            self.game.close()
            raise RuntimeError("Deadly Corridor does not provide the required movement buttons")
        self.game.new_episode()
        self.worker = threading.Thread(target=self._run_game, name="vizdoom-loop", daemon=True)
        self.worker.start()

    @staticmethod
    def _read_json(relative_path):
        path = ROOT / relative_path
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as file:
            return json.load(file)

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

    def request_reset(self):
        with self.lock:
            self.controls = {name: False for name in self.controls}
            self.reset_requested = True

    def status(self):
        with self.lock:
            current = dict(self.state)
        return {
            **current,
            "baseline_metrics": self.baseline_metrics,
            "attack_metrics": self.attack_metrics,
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
        return gray, model_input, prediction, confidence

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
                    controls = dict(self.controls)
                if reset or self.game.is_episode_finished():
                    self.game.new_episode()

                state = self.game.get_state()
                if state is None:
                    time.sleep(0.01)
                    continue
                gray, model_input, prediction, confidence = self._current_input(state)
                frame_jpeg = encode_image(gray)
                action = [0] * len(self.buttons)
                for name, pressed in controls.items():
                    button_index = self.button_indices[name]
                    if pressed and button_index is not None:
                        action[button_index] = 1
                self.game.make_action(action, 1)

                with self.lock:
                    self.state.update({
                        "ready": True,
                        "frame": frame_jpeg,
                        "prediction": prediction,
                        "confidence": confidence,
                        "fps": measured_fps,
                    })
                    self.current_model_input = model_input.copy()
                delay = (1.0 / self.target_fps) - (time.perf_counter() - tick_started)
                if delay > 0:
                    time.sleep(delay)
        except Exception as error:
            with self.lock:
                self.state["error"] = str(error)
                self.state["ready"] = False
        finally:
            self.game.close()

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

    demo = LiveDemo(str(model_path), args.fps)
    DashboardHandler.demo = demo
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), DashboardHandler)
    server.daemon_threads = True
    print(f"Dashboard: http://{host}:{args.port}")
    print("Controls: A/Left = turn left, D/Right = turn right, W/Up = forward, S/Down = backward, Space = fire")
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
