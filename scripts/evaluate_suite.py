"""Run comprehensive adversarial robustness analysis across models, attacks, and defenses.

Generates JSON artifacts in runs/analysis/ (and per-model PGD / v2 attack folders)
without modifying any original v1 official results (runs/baseline, runs/mitigation,
runs/fgsm, runs/fgsm_mitigated).

Outputs:
  - runs/pgd/attack_results.json (+ examples)         : PGD-10 on Baseline
  - runs/pgd_mitigated/attack_results.json           : PGD-10 on Mitigasi v1
  - runs/fgsm_mitigated_v2/attack_results.json       : FGSM on Mitigasi v2
  - runs/pgd_mitigated_v2/attack_results.json        : PGD-10 on Mitigasi v2
  - runs/analysis/robustness_curves.json             : ASR & adv_acc sweep over eps 0..0.10
  - runs/analysis/transferability.json               : Source -> Target black-box ASR matrix (eps=0.03)
  - runs/analysis/defense_comparison.json            : Input preprocessing vs Adv Training vs Defense-in-Depth
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    from scripts.adversarial_utils import (
        DEFENSE_LABELS,
        apply_defense,
        fgsm_attack,
        load_checkpoint_safe,
        pgd_attack,
    )
    from scripts.train_baseline import FrameDataset, SmallCNN
except ImportError:
    from adversarial_utils import (
        DEFENSE_LABELS,
        apply_defense,
        fgsm_attack,
        load_checkpoint_safe,
        pgd_attack,
    )
    from train_baseline import FrameDataset, SmallCNN


CURVE_EPSILONS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
EVAL_DEFENSES = ("none", "bit_depth_4", "bit_depth_3", "median_3x3", "gaussian_3x3", "bit4_median")


def load_model(path, device):
    model = SmallCNN().to(device)
    checkpoint = load_checkpoint_safe(path, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_all_test(data_dir, limit=0):
    ds = FrameDataset(os.path.join(data_dir, "split_test.csv"), data_dir, limit)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def batched_predict(model, x_all, defense="none", batch_size=64):
    preds = []
    with torch.no_grad():
        for start in range(0, x_all.size(0), batch_size):
            xb = apply_defense(x_all[start : start + batch_size], defense)
            preds.append(model(xb).argmax(1))
    return torch.cat(preds, dim=0)


def generate_adversarial_all(model, x_all, y_all, method, epsilon, pgd_steps=10, seed=12345, batch_size=64):
    if epsilon <= 0.0:
        return x_all.clone()
    gen = torch.Generator()
    gen.manual_seed(seed)
    advs = []
    for start in range(0, x_all.size(0), batch_size):
        xb = x_all[start : start + batch_size]
        yb = y_all[start : start + batch_size]
        if method == "pgd":
            x_adv = pgd_attack(
                model,
                xb,
                yb,
                epsilon=epsilon,
                steps=pgd_steps,
                random_start=True,
                criterion=F.cross_entropy,
                eval_mode=True,
                generator=gen,
            )
        else:
            x_adv = fgsm_attack(
                model,
                xb,
                yb,
                epsilon=epsilon,
                criterion=F.cross_entropy,
                eval_mode=True,
            )
        advs.append(x_adv)
    return torch.cat(advs, dim=0)


def compute_attack_metrics(y_true, clean_preds, adv_preds):
    y_np = y_true.cpu().numpy()
    cp_np = clean_preds.cpu().numpy()
    ap_np = adv_preds.cpu().numpy()
    was_correct = cp_np == y_np
    init_ok = int(was_correct.sum())
    init_ok_c0 = int((was_correct & (y_np == 0)).sum())
    init_ok_c1 = int((was_correct & (y_np == 1)).sum())
    flipped = was_correct & (ap_np != y_np)
    flips_total = int(flipped.sum())
    flips_1_to_0 = int((flipped & (y_np == 1)).sum())
    flips_0_to_1 = int((flipped & (y_np == 0)).sum())
    adv_correct = int((ap_np == y_np).sum())
    _, _, f1_clean, _ = precision_recall_fscore_support(y_np, cp_np, labels=[0, 1], zero_division=0)
    _, _, f1_adv, _ = precision_recall_fscore_support(y_np, ap_np, labels=[0, 1], zero_division=0)
    return {
        "test_n": int(len(y_np)),
        "clean_accuracy": float(accuracy_score(y_np, cp_np)),
        "clean_macro_f1": float(np.mean(f1_clean)),
        "initially_correct": init_ok,
        "initially_correct_class_0": init_ok_c0,
        "initially_correct_class_1": init_ok_c1,
        "flipped": flips_total,
        "asr": float(flips_total / init_ok) if init_ok else 0.0,
        "adv_accuracy": float(adv_correct / len(y_np)),
        "adv_macro_f1": float(np.mean(f1_adv)),
        "flips_1_to_0": flips_1_to_0,
        "flips_0_to_1": flips_0_to_1,
        "evasion_rate": float(flips_1_to_0 / init_ok_c1) if init_ok_c1 else 0.0,
        "false_alarm_rate": float(flips_0_to_1 / init_ok_c0) if init_ok_c0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data_final")
    parser.add_argument("--baseline", default="runs/baseline/best_model.pt")
    parser.add_argument("--mitigation", default="runs/mitigation/best_model.pt")
    parser.add_argument("--mitigation-v2", default="runs/mitigation_v2/best_model.pt")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--transfer-eps", type=float, default=0.03)
    parser.add_argument("--defense-eps", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="runs/analysis")
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cpu")
    x_test, y_test = load_all_test(args.data)
    print("Loaded test set:", tuple(x_test.shape))

    model_specs = [
        ("baseline", "Baseline", args.baseline),
        ("mitigation_v1", "Mitigasi v1 (FGSM 0.03)", args.mitigation),
        ("mitigation_v2", "Mitigasi v2 (FGSM-RS Multi-Eps)", args.mitigation_v2),
    ]
    models = {}
    labels = {}
    clean_preds_map = {}
    for key, label, path in model_specs:
        if os.path.isfile(path):
            model, ckpt = load_model(path, device)
            models[key] = model
            labels[key] = label
            clean_preds_map[key] = batched_predict(model, x_test, defense="none")
            print("Loaded %s (%s, epoch=%s)" % (key, path, ckpt.get("epoch")))
        else:
            print("Skipping missing model %s: %s" % (key, path))

    os.makedirs(args.out, exist_ok=True)

    # 1. Robustness Curves (FGSM & PGD-10 across CURVE_EPSILONS)
    print("\n--- 1. Robustness Curves ---")
    curves = {
        "epsilons": CURVE_EPSILONS,
        "pgd_steps": args.pgd_steps,
        "models": {},
    }
    cached_adv_transfer = {"fgsm": {}, "pgd": {}}
    for key, model in models.items():
        curves["models"][key] = {"label": labels[key], "fgsm": [], "pgd": []}
        for method in ("fgsm", "pgd"):
            for eps in CURVE_EPSILONS:
                x_adv = generate_adversarial_all(
                    model, x_test, y_test, method=method, epsilon=eps, pgd_steps=args.pgd_steps, seed=args.seed
                )
                if abs(eps - args.transfer_eps) < 1e-9:
                    cached_adv_transfer[method][key] = x_adv
                adv_preds = batched_predict(model, x_adv, defense="none")
                m = compute_attack_metrics(y_test, clean_preds_map[key], adv_preds)
                curves["models"][key][method].append({
                    "epsilon": eps,
                    "asr": m["asr"],
                    "adv_accuracy": m["adv_accuracy"],
                    "evasion_rate": m["evasion_rate"],
                    "false_alarm_rate": m["false_alarm_rate"],
                    "flips_1_to_0": m["flips_1_to_0"],
                    "flips_0_to_1": m["flips_0_to_1"],
                    "flipped": m["flipped"],
                    "denominator": m["initially_correct"],
                })
            print(
                "  %s [%s] eps=0.03 -> ASR=%.4f adv_acc=%.4f"
                % (
                    key,
                    method.upper(),
                    next(r["asr"] for r in curves["models"][key][method] if abs(r["epsilon"] - 0.03) < 1e-9),
                    next(r["adv_accuracy"] for r in curves["models"][key][method] if abs(r["epsilon"] - 0.03) < 1e-9),
                )
            )
    with open(os.path.join(args.out, "robustness_curves.json"), "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    # 2. Black-Box Transferability Matrix at transfer_eps
    print("\n--- 2. Transferability Matrix (eps=%.2f) ---" % args.transfer_eps)
    model_order = list(models.keys())
    transfer = {
        "epsilon": args.transfer_eps,
        "pgd_steps": args.pgd_steps,
        "models": [{"key": k, "label": labels[k]} for k in model_order],
        "fgsm": {},
        "pgd": {},
    }
    for method in ("fgsm", "pgd"):
        for src in model_order:
            x_adv_src = cached_adv_transfer[method][src]
            transfer[method][src] = {}
            for tgt in model_order:
                tgt_preds = batched_predict(models[tgt], x_adv_src, defense="none")
                m = compute_attack_metrics(y_test, clean_preds_map[tgt], tgt_preds)
                transfer[method][src][tgt] = {
                    "asr": m["asr"],
                    "adv_accuracy": m["adv_accuracy"],
                    "evasion_rate": m["evasion_rate"],
                    "false_alarm_rate": m["false_alarm_rate"],
                    "flipped": m["flipped"],
                    "denominator": m["initially_correct"],
                    "white_box": src == tgt,
                }
            row_str = "  ".join(
                "%s->%s: %.3f" % (src, tgt, transfer[method][src][tgt]["asr"]) for tgt in model_order
            )
            print("  [%s] %s" % (method.upper(), row_str))
    with open(os.path.join(args.out, "transferability.json"), "w", encoding="utf-8") as f:
        json.dump(transfer, f, indent=2)

    # 3. Defense-in-Depth Comparison (Input Preprocessing vs Adv Training vs Combined)
    print("\n--- 3. Input Preprocessing & Defense-in-Depth Comparison (eps=%.2f) ---" % args.defense_eps)
    defense_rows = []
    for key in model_order:
        model = models[key]
        x_adv_fgsm = cached_adv_transfer["fgsm"][key]
        x_adv_pgd = cached_adv_transfer["pgd"][key]
        for defense in EVAL_DEFENSES:
            clean_def = batched_predict(model, x_test, defense=defense)
            fgsm_def = batched_predict(model, x_adv_fgsm, defense=defense)
            pgd_def = batched_predict(model, x_adv_pgd, defense=defense)
            m_fgsm = compute_attack_metrics(y_test, clean_def, fgsm_def)
            m_pgd = compute_attack_metrics(y_test, clean_def, pgd_def)
            entry = {
                "model_key": key,
                "model_label": labels[key],
                "defense": defense,
                "defense_label": DEFENSE_LABELS[defense],
                "clean_accuracy": m_fgsm["clean_accuracy"],
                "clean_macro_f1": m_fgsm["clean_macro_f1"],
                "fgsm_asr": m_fgsm["asr"],
                "fgsm_adv_accuracy": m_fgsm["adv_accuracy"],
                "fgsm_evasion_rate": m_fgsm["evasion_rate"],
                "pgd_asr": m_pgd["asr"],
                "pgd_adv_accuracy": m_pgd["adv_accuracy"],
                "pgd_evasion_rate": m_pgd["evasion_rate"],
            }
            defense_rows.append(entry)
            print(
                "  %-13s + %-12s | clean=%.4f f1=%.4f | FGSM ASR=%.4f | PGD10 ASR=%.4f"
                % (
                    key,
                    defense,
                    entry["clean_accuracy"],
                    entry["clean_macro_f1"],
                    entry["fgsm_asr"],
                    entry["pgd_asr"],
                )
            )

    defense_report = {
        "epsilon": args.defense_eps,
        "pgd_steps": args.pgd_steps,
        "rows": defense_rows,
    }
    with open(os.path.join(args.out, "defense_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(defense_report, f, indent=2)

    print("\nAll analysis JSONs written to %s in %.1fs" % (args.out, time.time() - started))


if __name__ == "__main__":
    main()
