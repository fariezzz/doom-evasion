"""Automated test suite for DoomEvasion AI.

Run with:
    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import csv
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from live_demo import DashboardHandler, LiveDemo
from scripts.adversarial_utils import (
    DEFENSE_CHOICES,
    apply_defense,
    compute_gradcam,
    fgsm_attack,
    load_checkpoint_safe,
    perturbation_vis,
    pgd_attack,
    render_gradcam_overlay,
)
from scripts.train_baseline import SmallCNN


ROOT = Path(__file__).resolve().parent.parent


class TestDatasetIntegrity(unittest.TestCase):
    def test_splits_exist_and_no_episode_leakage(self):
        data_dir = ROOT / "data_final"
        split_eps = {}
        for split in ("train", "val", "test"):
            csv_path = data_dir / f"split_{split}.csv"
            self.assertTrue(csv_path.is_file(), f"Missing {csv_path}")
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            labels = {r["label"] for r in rows}
            self.assertEqual(labels, {"0", "1"}, f"Split {split} missing a class")
            split_eps[split] = {(r["episode"], r["seed"]) for r in rows}

        self.assertEqual(split_eps["train"] & split_eps["val"], set())
        self.assertEqual(split_eps["train"] & split_eps["test"], set())
        self.assertEqual(split_eps["val"] & split_eps["test"], set())


class TestModelAndAdversarialUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = SmallCNN().to(cls.device)
        ckpt_path = ROOT / "runs" / "baseline" / "best_model.pt"
        if ckpt_path.is_file():
            ckpt = load_checkpoint_safe(ckpt_path, cls.device)
            cls.model.load_state_dict(ckpt["state_dict"])
        cls.model.eval()

    def test_safe_checkpoint_loading(self):
        for rel in ("runs/baseline/best_model.pt", "runs/mitigation/best_model.pt"):
            path = ROOT / rel
            if path.is_file():
                ckpt = load_checkpoint_safe(path, self.device)
                self.assertIn("state_dict", ckpt)

    def test_smallcnn_output_shape(self):
        x = torch.rand(4, 1, 84, 84)
        logits = self.model(x)
        self.assertEqual(tuple(logits.shape), (4, 2))

    def test_fgsm_and_pgd_bounds(self):
        torch.manual_seed(42)
        x = torch.rand(2, 1, 84, 84)
        y = torch.tensor([0, 1], dtype=torch.long)
        eps = 0.03

        x_fgsm = fgsm_attack(self.model, x, y, epsilon=eps)
        self.assertEqual(tuple(x_fgsm.shape), tuple(x.shape))
        self.assertLessEqual(float((x_fgsm - x).abs().max()), eps + 1e-6)
        self.assertGreaterEqual(float(x_fgsm.min()), 0.0)
        self.assertLessEqual(float(x_fgsm.max()), 1.0)

        gen = torch.Generator().manual_seed(123)
        x_pgd = pgd_attack(self.model, x, y, epsilon=eps, steps=5, random_start=True, generator=gen)
        self.assertEqual(tuple(x_pgd.shape), tuple(x.shape))
        self.assertLessEqual(float((x_pgd - x).abs().max()), eps + 1e-6)
        self.assertGreaterEqual(float(x_pgd.min()), 0.0)
        self.assertLessEqual(float(x_pgd.max()), 1.0)

    def test_eval_mode_restored_after_attack(self):
        self.model.train()
        x = torch.rand(1, 1, 84, 84)
        y = torch.tensor([1], dtype=torch.long)
        _ = fgsm_attack(self.model, x, y, epsilon=0.03, eval_mode=True)
        self.assertTrue(self.model.training)
        self.model.eval()

    def test_all_defenses_preserve_shape_and_range(self):
        x = torch.rand(3, 1, 84, 84)
        for defense in DEFENSE_CHOICES:
            out = apply_defense(x, defense)
            self.assertEqual(tuple(out.shape), tuple(x.shape), f"Shape mismatch for {defense}")
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)

    def test_gradcam_and_overlay(self):
        x = torch.rand(1, 1, 84, 84)
        cam, logits = compute_gradcam(self.model, x)
        self.assertEqual(cam.shape, (84, 84))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)

        overlay = render_gradcam_overlay(x[0, 0].numpy(), cam)
        self.assertEqual(overlay.shape, (84, 84, 3))
        self.assertEqual(overlay.dtype, np.uint8)

        pert = perturbation_vis(x[0, 0].numpy(), x[0, 0].numpy(), gain=20.0)
        self.assertTrue(np.allclose(pert, 0.5))


class TestLiveDemoAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_path = ROOT / "runs" / "baseline" / "best_model.pt"
        mit1_path = ROOT / "runs" / "mitigation" / "best_model.pt"
        mit2_path = ROOT / "runs" / "mitigation_v2" / "best_model.pt"
        cls.demo = LiveDemo(
            str(base_path),
            fps=30.0,
            map_id="defend_the_center",
            mitigation_model_path=str(mit1_path) if mit1_path.is_file() else None,
            mitigation_v2_model_path=str(mit2_path) if mit2_path.is_file() else None,
            start_worker=False,
        )
        # Seed synthetic frame state so capture & evaluate endpoints can be tested without opening ViZDoom
        dummy_img = np.full((240, 320), 120, dtype=np.uint8)
        dummy_input = np.full((84, 84), 0.47, dtype=np.float32)
        from live_demo import encode_image
        with cls.demo.lock:
            cls.demo.state["ready"] = True
            cls.demo.state["frame"] = encode_image(dummy_img)
            cls.demo.state["prediction"] = 0
            cls.demo.state["confidence"] = 0.9
            cls.demo.current_model_input = dummy_input
            cls.demo.current_true_label = 0

        DashboardHandler.demo = cls.demo
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.demo.close()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_config_and_lean_status(self):
        code, cfg = self._get("/api/config")
        self.assertEqual(code, 200)
        self.assertIn("maps", cfg)
        self.assertIn("baseline_metrics", cfg)
        self.assertIn("defenses", cfg)
        self.assertTrue(cfg["mitigation_available"])
        self.assertTrue(cfg["mitigation_v2_available"])
        self.assertIsNotNone(cfg["robustness_curves"])
        self.assertIsNotNone(cfg["transferability"])
        self.assertIsNotNone(cfg["defense_comparison"])
        self.assertIsNotNone(cfg["pgd_metrics"])
        self.assertIsNotNone(cfg["pgd_mitigated_v2_metrics"])

        code, st = self._get("/api/status")
        self.assertEqual(code, 200)
        self.assertTrue(st["ready"])
        # Confirm static bloat is NOT sent in /api/status
        self.assertNotIn("baseline_metrics", st)
        self.assertNotIn("maps", st)

    def test_compare_mode_and_mitigation_v2(self):
        for mode in ("baseline", "mitigation", "mitigation_v2", "compare"):
            code, res = self._post("/api/model", {"mode": mode})
            self.assertEqual(code, 200)
            self.assertEqual(res["mode"], mode)
        self._post("/api/model", {"mode": "baseline"})

    def test_live_attack_and_view_endpoints(self):
        code, res = self._post("/api/live-attack", {"method": "fgsm", "epsilon": 0.03, "defense": "bit_depth_4"})
        self.assertEqual(code, 200)
        self.assertEqual(res["method"], "fgsm")
        self.assertEqual(res["defense"], "bit_depth_4")

        code, res = self._post("/api/view", {"overlay": "gradcam", "smoothing": True})
        self.assertEqual(code, 200)
        self.assertEqual(res["overlay"], "gradcam")
        self.assertTrue(res["smoothing"])

        # Reset back to clean
        self._post("/api/live-attack", {"method": "none", "epsilon": 0.03, "defense": "none"})
        self._post("/api/view", {"overlay": "clean", "smoothing": False})

    def test_capture_and_pgd_evaluate_with_xai(self):
        code, cap = self._post("/api/capture", {})
        self.assertEqual(code, 200)
        self.assertIn("frame", cap)

        code, ev = self._post("/api/evaluate-capture", {"label": 0, "attack": "pgd", "defense": "median_3x3"})
        self.assertEqual(code, 200)
        self.assertEqual(ev["attack"], "pgd")
        self.assertEqual(ev["defense"], "median_3x3")
        self.assertEqual(len(ev["baseline_samples"]), 4)
        sample = ev["baseline_samples"][1]
        self.assertIn("image", sample)
        self.assertIn("perturbation_image", sample)
        self.assertIn("gradcam_image", sample)
        self.assertIn("prob_monster", sample)
        self.assertIn("prob_no_monster", sample)

    def test_invalid_mode_returns_400(self):
        data = json.dumps({"mode": "nonexistent"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/model",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
