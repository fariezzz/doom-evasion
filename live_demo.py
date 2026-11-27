"""Run a local interactive ViZDoom and live CNN robustness demo.

Start from the project directory:
    py live_demo.py --model runs/baseline/best_model.pt --mitigation-model runs/mitigation/best_model.pt --mitigation-v2-model runs/mitigation_v2/best_model.pt --map defend_the_center

Features:
- Toggle Baseline / Mitigasi v1 / Mitigasi v2 / Bandingkan in real time.
- Live Adversarial Stream (FGSM / PGD-5) + Input Preprocessing Defenses (Bit-Depth, Median, Combo).
- Live Explainable AI (Grad-CAM heatmap overlay), Adversarial Input view, and Perturbation x20 view.
- Optional Temporal Smoothing (EMA) + Live Session Accuracy / Evasion telemetry.
- Multi-model single-frame Capture evaluation with FGSM / PGD-10, Grad-CAM, and Perturbation x20.
- Lean `/api/status` polling separated from static `/api/config` official & analysis metrics.
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

from scripts.adversarial_utils import (
    DEFENSE_CHOICES,
    DEFENSE_LABELS,
    apply_defense,
    compute_gradcam,
    fgsm_attack,
    load_checkpoint_safe,
    perturbation_vis,
    pgd_attack,
    render_gradcam_overlay,
)
from scripts.train_baseline import SmallCNN


ROOT = Path(__file__).resolve().parent
ALLOWED_EPSILONS = (0.01, 0.03, 0.05)
ALLOWED_LIVE_EPSILONS = (0.0, 0.01, 0.03, 0.05)
ALLOWED_LIVE_ATTACKS = ("none", "fgsm", "pgd")
ALLOWED_TARGET_SOURCES = ("whitebox", "transfer_baseline")
ALLOWED_OVERLAYS = ("clean", "adversarial", "gradcam", "perturbation")
ALLOWED_MODES = ("baseline", "mitigation", "mitigation_v2", "compare")
DEFAULT_MAP = "defend_the_center"
EMA_ALPHA = 0.35

ASSET_FOLDER_MAP = {
    "fgsm": "runs/fgsm",
    "fgsm-mitigated": "runs/fgsm_mitigated",
    "fgsm-mitigated-v2": "runs/fgsm_mitigated_v2",
    "pgd": "runs/pgd",
    "pgd-mitigated": "runs/pgd_mitigated",
    "pgd-mitigated-v2": "runs/pgd_mitigated_v2",
}

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
    """Ground-truth label from ViZDoom: 1 if any visible Monster, else 0."""
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


def encode_bgr_png(image_bgr):
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Could not encode a color evaluation image")
    return base64.b64encode(encoded).decode("ascii")


def _new_session_bucket():
    return {
        "frames": 0,
        "correct": 0,
        "monster_frames": 0,
        "evasions": 0,
        "safe_frames": 0,
        "false_alarms": 0,
    }


def _update_session_bucket(bucket, true_label, prediction):
    bucket["frames"] += 1
    if prediction == true_label:
        bucket["correct"] += 1
    if true_label == 1:
        bucket["monster_frames"] += 1
        if prediction == 0:
            bucket["evasions"] += 1
    else:
        bucket["safe_frames"] += 1
        if prediction == 1:
            bucket["false_alarms"] += 1


class LiveDemo:
    def __init__(self, model_path, fps, map_id, mitigation_model_path=None, mitigation_v2_model_path=None, start_worker=True):
        if map_id not in MAPS:
            raise ValueError("Unknown map: " + map_id)
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.model_path = model_path
        self.mitigation_model_path = mitigation_model_path
        self.mitigation_v2_model_path = mitigation_v2_model_path

        self.model = self._load_model(model_path)

        self.mitigation_model, self.mitigation_available, self.mitigation_error = self._load_optional_model(
            mitigation_model_path, "runs/mitigation/best_model.pt"
        )
        self.mitigation_v2_model, self.mitigation_v2_available, self.mitigation_v2_error = self._load_optional_model(
            mitigation_v2_model_path, "runs/mitigation_v2/best_model.pt"
        )

        self.model_mode = "baseline"
        self.live_attack_method = "none"
        self.live_attack_epsilon = 0.03
        self.live_attack_target_source = "whitebox"
        self.live_defense = "none"
        self.view_overlay = "clean"
        self.temporal_smoothing = False
        self.ema_probs = {"baseline": None, "mitigation": None, "mitigation_v2": None}
        self.session_stats = {
            "baseline": _new_session_bucket(),
            "mitigation": _new_session_bucket(),
            "mitigation_v2": _new_session_bucket(),
        }

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
            "prob_monster": None,
            "prob_no_monster": None,
            "true_label": None,
            "fps": 0.0,
            "map_id": map_id,
            "map_label": MAPS[map_id]["label"],
            "map_trained": MAPS[map_id]["trained"],
            "available_controls": {name: False for name in self.controls},
            "can_turn": False,
            "model_mode": "baseline",
            "live_attack_method": "none",
            "live_attack_epsilon": 0.03,
            "live_attack_target_source": "whitebox",
            "live_defense": "none",
            "view_overlay": "clean",
            "temporal_smoothing": False,
            "gradcam_model": None,
            "baseline_prediction": None,
            "baseline_confidence": None,
            "baseline_prob_monster": None,
            "mitigation_prediction": None,
            "mitigation_confidence": None,
            "mitigation_prob_monster": None,
            "mitigation_v2_prediction": None,
            "mitigation_v2_confidence": None,
            "mitigation_v2_prob_monster": None,
            "session_stats": self._copy_session_stats(),
        }
        self.baseline_capture = None
        self.capture_sequence = 0
        self.current_model_input = None
        self.current_true_label = None
        self._load_all_metrics()

        self.game = None
        self.buttons = []
        self.button_indices = {name: None for name in self.controls}
        self.turn_indices = {"left": None, "right": None}
        self.worker = None
        if start_worker:
            self.worker = threading.Thread(target=self._run_game, name="vizdoom-loop", daemon=True)
            self.worker.start()

    def _load_model(self, path):
        model = SmallCNN().to(self.device)
        checkpoint = load_checkpoint_safe(path, self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _load_optional_model(self, path_arg, label_path):
        if not path_arg:
            return None, False, f"Model dinonaktifkan ({label_path})"
        candidate = Path(path_arg)
        if not candidate.is_file():
            return None, False, f"Berkas model tidak ditemukan: {candidate}"
        try:
            model = self._load_model(str(candidate))
            return model, True, None
        except Exception as error:
            return None, False, str(error)

    def _load_all_metrics(self):
        self.baseline_metrics = self._read_json("runs/baseline/test_metrics.json")
        self.attack_metrics = self._read_json("runs/fgsm/attack_results.json")
        self.mitigation_metrics = self._read_json("runs/mitigation/test_metrics.json")
        self.mitigated_attack_metrics = self._read_json("runs/fgsm_mitigated/attack_results.json")
        self.mitigation_v2_metrics = self._read_json("runs/mitigation_v2/test_metrics.json")
        self.fgsm_mitigated_v2_metrics = self._read_json("runs/fgsm_mitigated_v2/attack_results.json")
        self.pgd_metrics = self._read_json("runs/pgd/attack_results.json")
        self.pgd_mitigated_metrics = self._read_json("runs/pgd_mitigated/attack_results.json")
        self.pgd_mitigated_v2_metrics = self._read_json("runs/pgd_mitigated_v2/attack_results.json")
        self.robustness_curves = self._read_json("runs/analysis/robustness_curves.json")
        self.transferability = self._read_json("runs/analysis/transferability.json")
        self.defense_comparison = self._read_json("runs/analysis/defense_comparison.json")

    def _copy_session_stats(self):
        return {k: dict(v) for k, v in self.session_stats.items()}

    def _reset_session_locked(self):
        self.ema_probs = {"baseline": None, "mitigation": None, "mitigation_v2": None}
        self.session_stats = {
            "baseline": _new_session_bucket(),
            "mitigation": _new_session_bucket(),
            "mitigation_v2": _new_session_bucket(),
        }
        self.state["session_stats"] = self._copy_session_stats()

    @staticmethod
    def _read_json(relative_path):
        path = ROOT / relative_path
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _build_game(map_id):
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

        button_indices = {
            "left": find("TURN_LEFT", "MOVE_LEFT"),
            "right": find("TURN_RIGHT", "MOVE_RIGHT"),
            "forward": find("MOVE_FORWARD"),
            "backward": find("MOVE_BACKWARD"),
            "attack": find("ATTACK"),
        }
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
            self._reset_session_locked()

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
            self._reset_session_locked()

    def request_reset(self):
        with self.lock:
            self.controls = {name: False for name in self.controls}
            self.turn_impulse = 0.0
            self.turn_clear_seq += 1
            self.reset_requested = True
            self._reset_session_locked()

    def reset_session_stats(self):
        with self.lock:
            self._reset_session_locked()
            return self._copy_session_stats()

    def set_model_mode(self, mode):
        if mode not in ALLOWED_MODES:
            raise ValueError("Mode model harus baseline, mitigation, mitigation_v2, atau compare")
        if mode == "mitigation" and not self.mitigation_available:
            raise ValueError(
                self.mitigation_error
                or "Model mitigasi v1 tidak tersedia. Periksa runs/mitigation/best_model.pt"
            )
        if mode == "mitigation_v2" and not self.mitigation_v2_available:
            raise ValueError(
                self.mitigation_v2_error
                or "Model mitigasi v2 tidak tersedia. Jalankan scripts/train_mitigation.py --v2"
            )
        if mode == "compare" and not (self.mitigation_available or self.mitigation_v2_available):
            raise ValueError("Minimal satu model mitigasi harus tersedia untuk mode Bandingkan")
        with self.lock:
            self.model_mode = mode
            self.state["model_mode"] = mode
            self._reset_session_locked()
        return mode

    def set_live_attack(self, method=None, epsilon=None, defense=None, target_source=None):
        with self.lock:
            new_method = self.live_attack_method if method is None else method
            new_eps = self.live_attack_epsilon if epsilon is None else epsilon
            new_def = self.live_defense if defense is None else defense
            new_src = self.live_attack_target_source if target_source is None else target_source
        if new_method not in ALLOWED_LIVE_ATTACKS:
            raise ValueError("Metode serangan live harus none, fgsm, atau pgd")
        if isinstance(new_eps, bool) or not isinstance(new_eps, (int, float)):
            raise ValueError("Epsilon harus berupa angka")
        new_eps = round(float(new_eps), 4)
        if new_eps not in ALLOWED_LIVE_EPSILONS:
            raise ValueError("Epsilon harus salah satu dari 0, 0.01, 0.03, atau 0.05")
        if new_def not in DEFENSE_CHOICES:
            raise ValueError("Pertahanan input tidak dikenali: " + str(new_def))
        if new_src not in ALLOWED_TARGET_SOURCES:
            raise ValueError("Sumber gradien harus whitebox atau transfer_baseline")
        with self.lock:
            self.live_attack_method = new_method
            self.live_attack_epsilon = new_eps
            self.live_defense = new_def
            self.live_attack_target_source = new_src
            self.state["live_attack_method"] = new_method
            self.state["live_attack_epsilon"] = new_eps
            self.state["live_defense"] = new_def
            self.state["live_attack_target_source"] = new_src
            self._reset_session_locked()
        return {
            "method": new_method,
            "epsilon": new_eps,
            "defense": new_def,
            "target_source": new_src,
        }

    def set_view_options(self, overlay=None, smoothing=None):
        with self.lock:
            new_overlay = self.view_overlay if overlay is None else overlay
            new_smoothing = self.temporal_smoothing if smoothing is None else smoothing
        if new_overlay not in ALLOWED_OVERLAYS:
            raise ValueError("Overlay harus clean, adversarial, gradcam, atau perturbation")
        if not isinstance(new_smoothing, bool):
            raise ValueError("Smoothing harus true atau false")
        with self.lock:
            self.view_overlay = new_overlay
            self.temporal_smoothing = new_smoothing
            self.state["view_overlay"] = new_overlay
            self.state["temporal_smoothing"] = new_smoothing
            self.ema_probs = {"baseline": None, "mitigation": None, "mitigation_v2": None}
        return {"overlay": new_overlay, "smoothing": new_smoothing}

    def config(self):
        """Static configuration and official/analysis evaluation artifacts (fetched once by dashboard)."""
        return {
            "maps": self.maps_info,
            "defenses": [{"id": k, "label": DEFENSE_LABELS[k]} for k in DEFENSE_CHOICES],
            "allowed_epsilons": list(ALLOWED_EPSILONS),
            "mitigation_available": self.mitigation_available,
            "mitigation_error": self.mitigation_error,
            "mitigation_v2_available": self.mitigation_v2_available,
            "mitigation_v2_error": self.mitigation_v2_error,
            "baseline_metrics": self.baseline_metrics,
            "attack_metrics": self.attack_metrics,
            "mitigation_metrics": self.mitigation_metrics,
            "mitigated_attack_metrics": self.mitigated_attack_metrics,
            "mitigation_v2_metrics": self.mitigation_v2_metrics,
            "fgsm_mitigated_v2_metrics": self.fgsm_mitigated_v2_metrics,
            "pgd_metrics": self.pgd_metrics,
            "pgd_mitigated_metrics": self.pgd_mitigated_metrics,
            "pgd_mitigated_v2_metrics": self.pgd_mitigated_v2_metrics,
            "robustness_curves": self.robustness_curves,
            "transferability": self.transferability,
            "defense_comparison": self.defense_comparison,
        }

    def status(self):
        """Lean dynamic state polled at ~30 Hz (no static JSON bloat)."""
        with self.lock:
            current = dict(self.state)
            current["session_stats"] = self._copy_session_stats()
        return {
            **current,
            "mitigation_available": self.mitigation_available,
            "mitigation_v2_available": self.mitigation_v2_available,
        }

    def capture_current_frame(self):
        with self.lock:
            if (
                not self.state["ready"]
                or self.state["frame"] is None
                or self.current_model_input is None
            ):
                raise RuntimeError(self.state["error"] or "Frame game belum tersedia")
            model_input_copy = self.current_model_input.copy()
            true_label = self.current_true_label
            frame_copy = self.state["frame"]
            active_mode = self.model_mode
            active_pred = self.state["prediction"]
            active_conf = self.state["confidence"]

        # Always compute clean predictions for all available models at capture time.
        tensor = torch.from_numpy(model_input_copy).unsqueeze(0).unsqueeze(0)
        with self.model_lock:
            base_info = self._predict_tensor(self.model, tensor)
            mit_info = self._predict_tensor(self.mitigation_model, tensor) if self.mitigation_available else None
            v2_info = self._predict_tensor(self.mitigation_v2_model, tensor) if self.mitigation_v2_available else None

        with self.lock:
            self.capture_sequence += 1
            self.baseline_capture = {
                "id": self.capture_sequence,
                "frame": frame_copy,
                "prediction": active_pred if active_pred is not None else base_info["prediction"],
                "confidence": active_conf if active_conf is not None else base_info["confidence"],
                "true_label": true_label,
                "model_input": model_input_copy,
                "evaluation": None,
                "baseline_prediction": base_info["prediction"],
                "baseline_confidence": base_info["confidence"],
                "baseline_prob_monster": base_info["prob_monster"],
                "mitigation_prediction": mit_info["prediction"] if mit_info else None,
                "mitigation_confidence": mit_info["confidence"] if mit_info else None,
                "mitigation_prob_monster": mit_info["prob_monster"] if mit_info else None,
                "mitigation_v2_prediction": v2_info["prediction"] if v2_info else None,
                "mitigation_v2_confidence": v2_info["confidence"] if v2_info else None,
                "mitigation_v2_prob_monster": v2_info["prob_monster"] if v2_info else None,
                "model_mode": active_mode,
                "mitigation_available": self.mitigation_available,
                "mitigation_v2_available": self.mitigation_v2_available,
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
            "baseline_prediction": capture.get("baseline_prediction"),
            "baseline_confidence": capture.get("baseline_confidence"),
            "baseline_prob_monster": capture.get("baseline_prob_monster"),
            "mitigation_prediction": capture.get("mitigation_prediction"),
            "mitigation_confidence": capture.get("mitigation_confidence"),
            "mitigation_prob_monster": capture.get("mitigation_prob_monster"),
            "mitigation_v2_prediction": capture.get("mitigation_v2_prediction"),
            "mitigation_v2_confidence": capture.get("mitigation_v2_confidence"),
            "mitigation_v2_prob_monster": capture.get("mitigation_v2_prob_monster"),
            "model_mode": capture.get("model_mode"),
            "mitigation_available": capture.get("mitigation_available"),
            "mitigation_v2_available": capture.get("mitigation_v2_available"),
        }

    def _evaluate_one_model(self, model, model_input, label, attack="fgsm", defense="none"):
        x = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).to(self.device)
        target = torch.tensor([label], dtype=torch.long, device=self.device)
        samples = []
        with self.model_lock:
            for eps in (0.0, *ALLOWED_EPSILONS):
                if eps == 0.0:
                    x_adv = x.clone()
                elif attack == "pgd":
                    gen = torch.Generator()
                    gen.manual_seed(12345)
                    x_adv = pgd_attack(
                        model, x, target, epsilon=eps, steps=10, random_start=True, eval_mode=True, generator=gen
                    )
                else:
                    x_adv = fgsm_attack(model, x, target, epsilon=eps, eval_mode=True)

                x_def = apply_defense(x_adv, defense)
                cam_01, logits = compute_gradcam(model, x_def)
                probs = torch.softmax(logits, dim=1)[0]
                prediction = int(probs.argmax().item())
                adv_np = x_def[0, 0].detach().cpu().numpy()
                pert_np = perturbation_vis(model_input, x_adv[0, 0].detach().cpu().numpy(), gain=20.0)
                gradcam_bgr = render_gradcam_overlay(adv_np, cam_01, alpha=0.48)
                samples.append({
                    "epsilon": eps,
                    "image": encode_gray_png(adv_np),
                    "perturbation_image": encode_gray_png(pert_np),
                    "gradcam_image": encode_bgr_png(gradcam_bgr),
                    "prediction": prediction,
                    "confidence": float(probs[prediction].item()),
                    "prob_no_monster": float(probs[0].item()),
                    "prob_monster": float(probs[1].item()),
                    "correct": prediction == label,
                })
        return samples

    def evaluate_baseline_capture(self, label, attack="fgsm", defense="none"):
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Label capture harus 0 (tanpa musuh) atau 1 (ada musuh)")
        if attack not in ("fgsm", "pgd"):
            raise ValueError("Metode serangan capture harus fgsm atau pgd")
        if defense not in DEFENSE_CHOICES:
            raise ValueError("Pertahanan input tidak dikenali: " + str(defense))

        with self.lock:
            if self.baseline_capture is None:
                raise RuntimeError("Ambil capture game terlebih dahulu")
            capture_id = self.baseline_capture["id"]
            model_input = self.baseline_capture["model_input"].copy()

        baseline_samples = self._evaluate_one_model(self.model, model_input, label, attack=attack, defense=defense)
        mitigation_samples = None
        if self.mitigation_available and self.mitigation_model is not None:
            mitigation_samples = self._evaluate_one_model(
                self.mitigation_model, model_input, label, attack=attack, defense=defense
            )
        mitigation_v2_samples = None
        if self.mitigation_v2_available and self.mitigation_v2_model is not None:
            mitigation_v2_samples = self._evaluate_one_model(
                self.mitigation_v2_model, model_input, label, attack=attack, defense=defense
            )

        result = {
            "label": label,
            "attack": attack,
            "defense": defense,
            "defense_label": DEFENSE_LABELS[defense],
            "samples": baseline_samples,
            "baseline_samples": baseline_samples,
            "mitigation_samples": mitigation_samples,
            "mitigation_v2_samples": mitigation_v2_samples,
            "mitigation_available": self.mitigation_available,
            "mitigation_v2_available": self.mitigation_v2_available,
            "mitigation_error": self.mitigation_error,
        }
        with self.lock:
            if self.baseline_capture is None or self.baseline_capture["id"] != capture_id:
                raise RuntimeError("Capture berubah saat evaluasi; ambil capture baru lalu ulangi")
            self.baseline_capture["evaluation"] = result
        return result

    def _predict_tensor(self, model, tensor):
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        prediction = int(probabilities.argmax().item())
        return {
            "prediction": prediction,
            "confidence": float(probabilities[prediction].item()),
            "prob_monster": float(probabilities[1].item()),
            "prob_no_monster": float(probabilities[0].item()),
        }

    def _apply_ema(self, key, info, smoothing_enabled):
        if info is None or not smoothing_enabled:
            return info
        prev = self.ema_probs.get(key)
        p_mon = info["prob_monster"]
        smoothed_mon = p_mon if prev is None else (EMA_ALPHA * p_mon + (1.0 - EMA_ALPHA) * prev)
        self.ema_probs[key] = smoothed_mon
        pred = 1 if smoothed_mon >= 0.5 else 0
        conf = smoothed_mon if pred == 1 else (1.0 - smoothed_mon)
        return {
            "prediction": pred,
            "confidence": float(conf),
            "prob_monster": float(smoothed_mon),
            "prob_no_monster": float(1.0 - smoothed_mon),
        }

    def _attack_tensor_live(self, model, tensor, target_tensor, method, epsilon):
        if method == "none" or epsilon <= 0.0:
            return tensor
        if method == "pgd":
            return pgd_attack(model, tensor, target_tensor, epsilon=epsilon, steps=5, random_start=True, eval_mode=True)
        return fgsm_attack(model, tensor, target_tensor, epsilon=epsilon, eval_mode=True)

    def _current_input(self, state):
        gray = to_gray_frame(state.screen_buffer).copy()
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        model_input = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).to(self.device)
        true_label = true_label_from_state(state)
        target_tensor = torch.tensor([true_label], dtype=torch.long, device=self.device)

        with self.lock:
            mode = self.model_mode
            attack_method = self.live_attack_method
            attack_eps = self.live_attack_epsilon
            target_source = self.live_attack_target_source
            defense = self.live_defense
            overlay = self.view_overlay
            smoothing = self.temporal_smoothing

        need_baseline = mode in ("baseline", "compare") or (
            attack_method != "none" and attack_eps > 0.0 and target_source == "transfer_baseline"
        )
        need_mit1 = self.mitigation_available and mode in ("mitigation", "compare")
        need_mit2 = self.mitigation_v2_available and mode in ("mitigation_v2", "compare")

        baseline_info = None
        mit1_info = None
        mit2_info = None
        primary_adv = tensor
        primary_def = apply_defense(tensor, defense)
        primary_model = self.model
        gradcam_model_name = "baseline"
        cam_01 = None

        with self.model_lock:
            shared_adv = None
            if attack_method != "none" and attack_eps > 0.0 and target_source == "transfer_baseline":
                shared_adv = self._attack_tensor_live(self.model, tensor, target_tensor, attack_method, attack_eps)

            if need_baseline:
                x_adv_b = shared_adv if shared_adv is not None else self._attack_tensor_live(
                    self.model, tensor, target_tensor, attack_method, attack_eps
                )
                x_def_b = apply_defense(x_adv_b, defense)
                if mode in ("baseline", "compare"):
                    baseline_info = self._predict_tensor(self.model, x_def_b)
                    primary_adv = x_adv_b
                    primary_def = x_def_b
                    primary_model = self.model
                    gradcam_model_name = "baseline"

            if need_mit1:
                x_adv_m1 = shared_adv if shared_adv is not None else self._attack_tensor_live(
                    self.mitigation_model, tensor, target_tensor, attack_method, attack_eps
                )
                x_def_m1 = apply_defense(x_adv_m1, defense)
                mit1_info = self._predict_tensor(self.mitigation_model, x_def_m1)
                if mode == "mitigation":
                    primary_adv = x_adv_m1
                    primary_def = x_def_m1
                    primary_model = self.mitigation_model
                    gradcam_model_name = "mitigation"

            if need_mit2:
                x_adv_m2 = shared_adv if shared_adv is not None else self._attack_tensor_live(
                    self.mitigation_v2_model, tensor, target_tensor, attack_method, attack_eps
                )
                x_def_m2 = apply_defense(x_adv_m2, defense)
                mit2_info = self._predict_tensor(self.mitigation_v2_model, x_def_m2)
                if mode == "mitigation_v2":
                    primary_adv = x_adv_m2
                    primary_def = x_def_m2
                    primary_model = self.mitigation_v2_model
                    gradcam_model_name = "mitigation_v2"

            if overlay == "gradcam":
                cam_01, _ = compute_gradcam(primary_model, primary_def)

        baseline_info = self._apply_ema("baseline", baseline_info, smoothing)
        mit1_info = self._apply_ema("mitigation", mit1_info, smoothing)
        mit2_info = self._apply_ema("mitigation_v2", mit2_info, smoothing)

        if mode == "mitigation" and mit1_info is not None:
            active = mit1_info
        elif mode == "mitigation_v2" and mit2_info is not None:
            active = mit2_info
        else:
            active = baseline_info

        # Render display frame according to selected overlay
        h_disp, w_disp = gray.shape[:2]
        if overlay == "gradcam" and cam_01 is not None:
            display_frame = render_gradcam_overlay(gray, cam_01, alpha=0.45)
        elif overlay == "adversarial":
            adv_u8 = np.rint(np.clip(primary_def[0, 0].cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
            display_frame = cv2.resize(adv_u8, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)
        elif overlay == "perturbation":
            pert_np = perturbation_vis(model_input, primary_adv[0, 0].cpu().numpy(), gain=20.0)
            pert_u8 = np.rint(pert_np * 255.0).astype(np.uint8)
            display_frame = cv2.resize(pert_u8, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)
        else:
            display_frame = gray

        return (
            display_frame,
            model_input,
            active,
            true_label,
            baseline_info,
            mit1_info,
            mit2_info,
            gradcam_model_name if overlay == "gradcam" else None,
        )

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
                (
                    display_frame,
                    model_input,
                    active,
                    true_label,
                    baseline_info,
                    mit1_info,
                    mit2_info,
                    gradcam_model_name,
                ) = self._current_input(state)
                frame_jpeg = encode_image(display_frame)
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
                    if baseline_info is not None:
                        _update_session_bucket(self.session_stats["baseline"], true_label, baseline_info["prediction"])
                    if mit1_info is not None:
                        _update_session_bucket(self.session_stats["mitigation"], true_label, mit1_info["prediction"])
                    if mit2_info is not None:
                        _update_session_bucket(self.session_stats["mitigation_v2"], true_label, mit2_info["prediction"])

                    self.state.update({
                        "ready": True,
                        "error": None,
                        "frame": frame_jpeg,
                        "prediction": active["prediction"] if active else None,
                        "confidence": active["confidence"] if active else None,
                        "prob_monster": active["prob_monster"] if active else None,
                        "prob_no_monster": active["prob_no_monster"] if active else None,
                        "true_label": true_label,
                        "fps": measured_fps,
                        "model_mode": self.model_mode,
                        "live_attack_method": self.live_attack_method,
                        "live_attack_epsilon": self.live_attack_epsilon,
                        "live_attack_target_source": self.live_attack_target_source,
                        "live_defense": self.live_defense,
                        "view_overlay": self.view_overlay,
                        "temporal_smoothing": self.temporal_smoothing,
                        "gradcam_model": gradcam_model_name,
                        "baseline_prediction": baseline_info["prediction"] if baseline_info else None,
                        "baseline_confidence": baseline_info["confidence"] if baseline_info else None,
                        "baseline_prob_monster": baseline_info["prob_monster"] if baseline_info else None,
                        "mitigation_prediction": mit1_info["prediction"] if mit1_info else None,
                        "mitigation_confidence": mit1_info["confidence"] if mit1_info else None,
                        "mitigation_prob_monster": mit1_info["prob_monster"] if mit1_info else None,
                        "mitigation_v2_prediction": mit2_info["prediction"] if mit2_info else None,
                        "mitigation_v2_confidence": mit2_info["confidence"] if mit2_info else None,
                        "mitigation_v2_prob_monster": mit2_info["prob_monster"] if mit2_info else None,
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
        if self.worker is not None:
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
        if path == "/api/config":
            return self._send_json(self.demo.config())
        if path == "/api/status":
            return self._send_json(self.demo.status())
        if path == "/api/capture":
            return self._send_json(self.demo.get_baseline_capture() or {})
        if path.startswith("/assets/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[1] in ASSET_FOLDER_MAP:
                name = parts[2]
                allowed_names = {f"examples_eps_{epsilon:.2f}.png" for epsilon in ALLOWED_EPSILONS}
                if name in allowed_names:
                    self.path = "/" + ASSET_FOLDER_MAP[parts[1]] + "/" + name
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
                    raise ValueError('Kirim delta mouse: {"delta": angka}')
                if "cancel" in body and not isinstance(body["cancel"], bool):
                    raise ValueError("Cancel harus true atau false")
                if body.get("cancel") is True:
                    self.demo.cancel_turn()
                    return self._send_json({"ok": True})
                if "delta" not in body:
                    raise ValueError('Kirim delta mouse: {"delta": angka}')
                self.demo.add_turn(body["delta"])
                return self._send_json({"ok": True})
            if path == "/api/model":
                if not isinstance(body, dict) or set(body) != {"mode"}:
                    raise ValueError('Kirim mode model: {"mode": "baseline"|"mitigation"|"mitigation_v2"|"compare"}')
                mode = self.demo.set_model_mode(body["mode"])
                return self._send_json({"ok": True, "mode": mode})
            if path == "/api/live-attack":
                if not isinstance(body, dict) or set(body) - {"method", "epsilon", "defense", "target_source"}:
                    raise ValueError("Parameter live-attack tidak valid")
                result = self.demo.set_live_attack(
                    method=body.get("method"),
                    epsilon=body.get("epsilon"),
                    defense=body.get("defense"),
                    target_source=body.get("target_source"),
                )
                return self._send_json({"ok": True, **result})
            if path == "/api/view":
                if not isinstance(body, dict) or set(body) - {"overlay", "smoothing"}:
                    raise ValueError("Parameter view tidak valid")
                result = self.demo.set_view_options(
                    overlay=body.get("overlay"),
                    smoothing=body.get("smoothing"),
                )
                return self._send_json({"ok": True, **result})
            if path == "/api/reset-stats":
                return self._send_json({"ok": True, "session_stats": self.demo.reset_session_stats()})
            if path == "/api/map":
                if not isinstance(body, dict) or set(body) != {"map"}:
                    raise ValueError('Kirim id map: {"map": "defend_the_center"}')
                self.demo.request_map(body["map"])
                return self._send_json({"ok": True})
            if path == "/api/reset":
                self.demo.request_reset()
                return self._send_json({"ok": True})
            if path == "/api/capture":
                return self._send_json(self.demo.capture_current_frame())
            if path == "/api/evaluate-capture":
                if not isinstance(body, dict) or "label" not in body or set(body) - {"label", "attack", "defense"}:
                    raise ValueError("Kirim label capture: 0 atau 1 (opsional attack & defense)")
                return self._send_json(
                    self.demo.evaluate_baseline_capture(
                        body["label"],
                        attack=body.get("attack", "fgsm"),
                        defense=body.get("defense", "none"),
                    )
                )
            self.send_error(404, "Not found")
        except (ValueError, RuntimeError) as error:
            self._send_json({"error": str(error)}, status=400)


def _resolve_path(path_arg):
    p = Path(path_arg)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/baseline/best_model.pt")
    parser.add_argument("--mitigation-model", default="runs/mitigation/best_model.pt")
    parser.add_argument("--mitigation-v2-model", default="runs/mitigation_v2/best_model.pt")
    parser.add_argument("--no-mitigation-model", action="store_true", help="Jangan muat model mitigasi")
    parser.add_argument("--map", dest="map_id", default=DEFAULT_MAP, choices=sorted(MAPS))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.fps < 1 or args.fps > 60:
        parser.error("--fps must be between 1 and 60")

    model_path = _resolve_path(args.model)
    if not model_path.is_file():
        parser.error("Model checkpoint not found: " + str(model_path))

    mitigation_path = None
    mitigation_v2_path = None
    if not args.no_mitigation_model:
        mit1 = _resolve_path(args.mitigation_model)
        if mit1.is_file():
            mitigation_path = str(mit1)
        else:
            print("Peringatan: model mitigasi v1 tidak ditemukan: " + str(mit1))
        mit2 = _resolve_path(args.mitigation_v2_model)
        if mit2.is_file():
            mitigation_v2_path = str(mit2)
        else:
            print("Info: model mitigasi v2 belum tersedia: " + str(mit2))

    demo = LiveDemo(
        str(model_path),
        args.fps,
        args.map_id,
        mitigation_model_path=mitigation_path,
        mitigation_v2_model_path=mitigation_v2_path,
    )
    DashboardHandler.demo = demo
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), DashboardHandler)
    server.daemon_threads = True
    print(f"Dashboard: http://{host}:{args.port}")
    print(f"Map: {MAPS[args.map_id]['label']} (model dilatih pada Defend the Center)")
    print("Model mitigasi v1:", "tersedia (" + str(demo.mitigation_model_path) + ")" if demo.mitigation_available else "tidak tersedia")
    print("Model mitigasi v2:", "tersedia (" + str(demo.mitigation_v2_model_path) + ")" if demo.mitigation_v2_available else "belum tersedia")
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
