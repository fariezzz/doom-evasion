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
from torch.utils.data import DataLoader

try:
    from scripts.adversarial_utils import (
        DEFENSE_CHOICES,
        apply_defense,
        fgsm_attack,
        load_checkpoint_safe,
        perturbation_vis,
        pgd_attack,
    )
    from scripts.train_baseline import FrameDataset, SmallCNN
except ImportError:
    from adversarial_utils import (
        DEFENSE_CHOICES,
        apply_defense,
        fgsm_attack,
        load_checkpoint_safe,
        perturbation_vis,
        pgd_attack,
    )
    from train_baseline import FrameDataset, SmallCNN


def save_examples(examples, eps, attack_name, out_path):
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row, (clean, adv, info) in enumerate(examples):
        pert = perturbation_vis(clean, adv, gain=20.0)
        for col, (img, title) in enumerate(
            ((clean, "bersih"), (adv, "%s eps=%s" % (attack_name.upper(), eps)), (pert, "perturbasi x20"))
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
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha", type=float, default=None)
    parser.add_argument("--defense", choices=list(DEFENSE_CHOICES), default="none")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="runs/fgsm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print("torch:", torch.__version__, "| device:", device, "| attack:", args.attack, "| defense:", args.defense)

    model = SmallCNN()
    checkpoint = load_checkpoint_safe(args.model, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Model dimuat dari:", args.model, "| epoch checkpoint:", checkpoint.get("epoch"))

    test_ds = FrameDataset(os.path.join(args.data, "split_test.csv"), args.data, args.limit)
    print("Test: %d frame" % len(test_ds))
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    per_eps = {
        eps: {
            "flipped": 0,
            "adv_correct": 0,
            "flips_1_to_0": 0,
            "flips_0_to_1": 0,
            "examples": [],
        }
        for eps in args.epsilons
    }
    rows = []
    initially_correct = 0
    initially_correct_c0 = 0
    initially_correct_c1 = 0
    clean_correct = 0
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    started = time.time()
    offset = 0
    for x_batch, y_batch in loader:
        bsz = x_batch.size(0)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            x_clean_def = apply_defense(x_batch, args.defense)
            clean_preds = model(x_clean_def).argmax(1).cpu().tolist()

        batch_rows = []
        y_list = y_batch.cpu().tolist()
        for i in range(bsz):
            path = test_ds.items[offset + i][0]
            y_val = y_list[i]
            cp = clean_preds[i]
            was_ok = cp == y_val
            if was_ok:
                initially_correct += 1
                clean_correct += 1
                if y_val == 1:
                    initially_correct_c1 += 1
                else:
                    initially_correct_c0 += 1
            batch_rows.append({"path": os.path.relpath(path, args.data), "label": y_val, "clean_pred": cp})

        for eps in args.epsilons:
            if args.attack == "pgd":
                x_adv = pgd_attack(
                    model,
                    x_batch,
                    y_batch,
                    epsilon=eps,
                    steps=args.pgd_steps,
                    alpha=args.pgd_alpha,
                    random_start=True,
                    criterion=F.cross_entropy,
                    eval_mode=True,
                    generator=gen,
                )
            else:
                x_adv = fgsm_attack(
                    model,
                    x_batch,
                    y_batch,
                    epsilon=eps,
                    criterion=F.cross_entropy,
                    eval_mode=True,
                )
            with torch.no_grad():
                x_adv_def = apply_defense(x_adv, args.defense)
                adv_preds = model(x_adv_def).argmax(1).cpu().tolist()

            stats = per_eps[eps]
            for i in range(bsz):
                y_val = y_list[i]
                cp = clean_preds[i]
                ap = adv_preds[i]
                was_ok = cp == y_val
                flipped = was_ok and ap != y_val
                if ap == y_val:
                    stats["adv_correct"] += 1
                if flipped:
                    stats["flipped"] += 1
                    if y_val == 1:
                        stats["flips_1_to_0"] += 1
                    else:
                        stats["flips_0_to_1"] += 1
                    if len(stats["examples"]) < args.examples:
                        clean_np = x_batch[i, 0].cpu().numpy()
                        adv_np = x_adv_def[i, 0].cpu().numpy()
                        info = "label %d: %d -> %d" % (y_val, cp, ap)
                        stats["examples"].append((clean_np, adv_np, info))
                batch_rows[i]["adv_pred_eps_%s" % eps] = ap
                batch_rows[i]["flipped_eps_%s" % eps] = int(flipped)

        rows.extend(batch_rows)
        offset += bsz

    elapsed = time.time() - started
    clean_accuracy = clean_correct / len(test_ds)
    expected_note = ""
    metrics_candidate = os.path.join(os.path.dirname(args.model), "test_metrics.json")
    if args.defense == "none" and os.path.isfile(metrics_candidate):
        try:
            with open(metrics_candidate, encoding="utf-8") as mf:
                m_acc = json.load(mf).get("test", {}).get("accuracy")
            if isinstance(m_acc, (int, float)):
                expected_note = " (test_metrics model: %.4f)" % m_acc
        except Exception:
            pass
    print("Akurasi bersih test: %.4f%s" % (clean_accuracy, expected_note))
    print(
        "Sampel awalnya benar: %d dari %d (kelas 0=%d, kelas 1=%d)"
        % (initially_correct, len(test_ds), initially_correct_c0, initially_correct_c1)
    )

    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.out):
        if name.startswith("examples_eps_") and name.endswith(".png"):
            os.remove(os.path.join(args.out, name))

    results = {
        "model": args.model,
        "attack": args.attack,
        "pgd_steps": args.pgd_steps if args.attack == "pgd" else 1,
        "defense": args.defense,
        "data": args.data,
        "test_n": len(test_ds),
        "initially_correct": initially_correct,
        "initially_correct_class_0": initially_correct_c0,
        "initially_correct_class_1": initially_correct_c1,
        "clean_accuracy": clean_accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "per_epsilon": [],
    }
    for eps in args.epsilons:
        stats = per_eps[eps]
        asr = stats["flipped"] / initially_correct if initially_correct else 0.0
        adv_accuracy = stats["adv_correct"] / len(test_ds)
        evasion_rate = stats["flips_1_to_0"] / initially_correct_c1 if initially_correct_c1 else 0.0
        false_alarm_rate = stats["flips_0_to_1"] / initially_correct_c0 if initially_correct_c0 else 0.0
        entry = {
            "epsilon": eps,
            "asr": asr,
            "flipped": stats["flipped"],
            "denominator": initially_correct,
            "adv_accuracy": adv_accuracy,
            "flips_1_to_0": stats["flips_1_to_0"],
            "flips_0_to_1": stats["flips_0_to_1"],
            "evasion_rate": evasion_rate,
            "false_alarm_rate": false_alarm_rate,
            "examples_saved": len(stats["examples"]),
        }
        results["per_epsilon"].append(entry)
        print(
            "eps=%s ASR=%.4f (%d/%d) adv_acc=%.4f evasion(1->0)=%.4f(%d) false_alarm(0->1)=%.4f(%d)"
            % (
                eps,
                asr,
                stats["flipped"],
                initially_correct,
                adv_accuracy,
                evasion_rate,
                stats["flips_1_to_0"],
                false_alarm_rate,
                stats["flips_0_to_1"],
            )
        )
        if stats["examples"]:
            save_examples(
                stats["examples"],
                eps,
                args.attack,
                os.path.join(args.out, "examples_eps_%s.png" % eps),
            )
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
