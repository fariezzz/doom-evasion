"""Serangan Fast Gradient Sign Method (FGSM) ke CNN baseline DoomEvasion (Langkah 4).

Memuat model terbaik dari runs/baseline/best_model.pt, menyerang tiap
frame test dengan FGSM pada 3 epsilon, dan menghitung Attack Success
Rate (ASR) hanya dari sampel test yang awalnya diprediksi benar.
Gradien dihitung sekali per sampel dari frame bersih, lalu dipakai
untuk ketiga epsilon (FGSM standar: gradien tidak bergantung epsilon).
Rumus: x_adv = clip(x + epsilon * sign(gradien), 0, 1), skala 0 sampai 1.
Jalankan dari folder doom-evasion dengan venv aktif.
Contoh run penuh:
    python attack_fgsm.py --data data_final --model runs/baseline/best_model.pt --out runs/fgsm
Contoh smoke test (verifikasi script jalan, BUKAN hasil serangan):
    python attack_fgsm.py --data data_final --model runs/baseline/best_model.pt --limit 32 --out runs/fgsm_smoke
Output di folder --out: attack_results.json, attack_log.csv, dan
gambar contoh examples_eps_<nilai>.png per epsilon (baris = sampel,
kolom = bersih, adversarial, perturbasi yang diperkuat 20x).
"""

import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from train_baseline import FrameDataset, SmallCNN


def save_examples(examples, eps, out_path):
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row, (clean, adv, info) in enumerate(examples):
        pert = np.clip(0.5 + (adv - clean) * 20.0, 0.0, 1.0)
        for col, (img, title) in enumerate(
            ((clean, "bersih"), (adv, "adversarial eps=%s" % eps), (pert, "perturbasi x20"))
        ):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel(info, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_final")
    parser.add_argument("--model", default="runs/baseline/best_model.pt")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="runs/fgsm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print("torch:", torch.__version__, "| device:", device)

    model = SmallCNN()
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Model dimuat dari:", args.model, "| epoch checkpoint:", checkpoint.get("epoch"))

    test_ds = FrameDataset(os.path.join(args.data, "split_test.csv"), args.data, args.limit)
    print("Test: %d frame" % len(test_ds))

    criterion = F.cross_entropy
    rows = []
    per_eps = {eps: {"flipped": 0, "adv_correct": 0, "flips_1_to_0": 0, "flips_0_to_1": 0, "examples": []} for eps in args.epsilons}
    initially_correct = 0
    clean_correct = 0
    started = time.time()
    for idx in range(len(test_ds)):
        x, y = test_ds[idx]
        path = test_ds.items[idx][0]
        x = x.unsqueeze(0).to(device)
        y_tensor = torch.tensor([y], dtype=torch.long, device=device)
        x_req = x.clone().detach().requires_grad_(True)
        logits = model(x_req)
        loss = criterion(logits, y_tensor)
        model.zero_grad()
        loss.backward()
        grad = x_req.grad.detach()
        clean_pred = int(logits.argmax(1).item())
        was_correct = clean_pred == y
        if was_correct:
            initially_correct += 1
            clean_correct += 1
        row = {"path": os.path.relpath(path, args.data), "label": y, "clean_pred": clean_pred}
        clean_np = x.squeeze(0).squeeze(0).cpu().numpy()
        for eps in args.epsilons:
            x_adv = torch.clamp(x + eps * torch.sign(grad), 0.0, 1.0)
            with torch.no_grad():
                adv_pred = int(model(x_adv).argmax(1).item())
            flipped = was_correct and adv_pred != y
            stats = per_eps[eps]
            if adv_pred == y:
                stats["adv_correct"] += 1
            if flipped:
                stats["flipped"] += 1
                if y == 1:
                    stats["flips_1_to_0"] += 1
                else:
                    stats["flips_0_to_1"] += 1
                if len(stats["examples"]) < args.examples:
                    adv_np = x_adv.squeeze(0).squeeze(0).cpu().numpy()
                    info = "label %d: %d -> %d" % (y, clean_pred, adv_pred)
                    stats["examples"].append((clean_np, adv_np, info))
            row["adv_pred_eps_%s" % eps] = adv_pred
            row["flipped_eps_%s" % eps] = int(flipped)
        rows.append(row)
    elapsed = time.time() - started
    clean_accuracy = clean_correct / len(test_ds)
    print("Akurasi bersih test: %.4f (harusnya cocok dengan baseline 0.9628)" % clean_accuracy)
    print("Sampel awalnya benar: %d dari %d" % (initially_correct, len(test_ds)))

    os.makedirs(args.out, exist_ok=True)
    results = {
        "model": args.model,
        "data": args.data,
        "test_n": len(test_ds),
        "initially_correct": initially_correct,
        "clean_accuracy": clean_accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "per_epsilon": [],
    }
    for eps in args.epsilons:
        stats = per_eps[eps]
        asr = stats["flipped"] / initially_correct if initially_correct else 0.0
        adv_accuracy = stats["adv_correct"] / len(test_ds)
        entry = {
            "epsilon": eps,
            "asr": asr,
            "flipped": stats["flipped"],
            "denominator": initially_correct,
            "adv_accuracy": adv_accuracy,
            "flips_1_to_0": stats["flips_1_to_0"],
            "flips_0_to_1": stats["flips_0_to_1"],
            "examples_saved": len(stats["examples"]),
        }
        results["per_epsilon"].append(entry)
        print("eps=%s ASR=%.4f (%d/%d) adv_acc=%.4f flip 1->0=%d flip 0->1=%d contoh=%d"
              % (eps, asr, stats["flipped"], initially_correct, adv_accuracy,
                 stats["flips_1_to_0"], stats["flips_0_to_1"], len(stats["examples"])))
        if stats["examples"]:
            save_examples(stats["examples"], eps, os.path.join(args.out, "examples_eps_%s.png" % eps))
    with open(os.path.join(args.out, "attack_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    fieldnames = ["path", "label", "clean_pred"]
    for eps in args.epsilons:
        fieldnames += ["adv_pred_eps_%s" % eps, "flipped_eps_%s" % eps]
    with open(os.path.join(args.out, "attack_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Selesai dalam %.1f detik. Output tersimpan di: %s" % (elapsed, args.out))


if __name__ == "__main__":
    main()
