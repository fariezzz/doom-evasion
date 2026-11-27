import argparse
import csv
import json
import os
import random
import time
from collections import Counter

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    from scripts.adversarial_utils import fgsm_attack, load_checkpoint_safe, pgd_attack
    from scripts.train_baseline import FrameDataset, SmallCNN, evaluate, save_confusion_matrix
except ImportError:
    from adversarial_utils import fgsm_attack, load_checkpoint_safe, pgd_attack
    from train_baseline import FrameDataset, SmallCNN, evaluate, save_confusion_matrix


def train_epoch_adv(
    model,
    loader,
    criterion,
    optimizer,
    device,
    eps_choices,
    eval_mode_adv=False,
    random_start=False,
    adv_method="fgsm",
    pgd_steps=3,
    rng=None,
    torch_gen=None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_eps = eps_choices[0] if len(eps_choices) == 1 else (
            rng.choice(eps_choices) if rng is not None else random.choice(eps_choices)
        )
        # 1. Buat contoh adversarial (opsional eval_mode untuk mematikan Dropout saat hitung gradien).
        if adv_method == "pgd":
            x_adv = pgd_attack(
                model,
                x,
                y,
                epsilon=batch_eps,
                steps=pgd_steps,
                random_start=random_start,
                criterion=criterion,
                eval_mode=eval_mode_adv,
                generator=torch_gen,
            )
        else:
            x_adv = fgsm_attack(
                model,
                x,
                y,
                epsilon=batch_eps,
                criterion=criterion,
                eval_mode=eval_mode_adv,
                random_start=random_start,
                generator=torch_gen,
            )

        # 2. Satu langkah optimizer dari rata-rata loss bersih dan adversarial.
        model.train()
        optimizer.zero_grad()
        logits_clean = model(x)
        loss_clean = criterion(logits_clean, y)
        loss_adv = criterion(model(x_adv), y)
        loss = (loss_clean + loss_adv) / 2.0
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits_clean.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate_adversarial(model, loader, criterion, device, epsilon):
    """Evaluate model robustness on a validation/test DataLoader under FGSM at epsilon."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_true = []
    all_pred = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, epsilon=epsilon, criterion=criterion, eval_mode=True)
        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(all_true, all_pred),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": cm.tolist(),
        "n": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_final")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train-eps", type=float, default=0.03)
    parser.add_argument("--train-eps-list", type=float, nargs="+", default=None)
    parser.add_argument("--adv-method", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--eval-mode-adv", action="store_true", help="Matikan dropout saat menghitung gradien adversarial")
    parser.add_argument("--random-start", action="store_true", help="Gunakan random start L_inf (FGSM-RS / PGD)")
    parser.add_argument("--robust-val-weight", type=float, default=0.0, help="Bobot val_adv_macro_f1 (0..1) untuk memilih checkpoint terbaik")
    parser.add_argument("--v2", action="store_true", help="Preset mitigasi v2: FGSM-RS multi-eps, eval-mode-adv, dan seleksi checkpoint gabungan bersih+robust ke runs/mitigation_v2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.v2:
        args.eval_mode_adv = True
        args.random_start = True
        if args.train_eps_list is None:
            args.train_eps_list = [0.01, 0.03, 0.05]
        if args.robust_val_weight == 0.0:
            args.robust_val_weight = 0.5
        if args.out is None:
            args.out = "runs/mitigation_v2"
    else:
        if args.out is None:
            args.out = "runs/mitigation"

    eps_choices = args.train_eps_list if args.train_eps_list else [args.train_eps]
    val_eval_eps = args.train_eps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    adv_gen = torch.Generator()
    adv_gen.manual_seed(args.seed + 999)

    device = torch.device("cpu")
    print("torch:", torch.__version__, "| device:", device, "| threads:", torch.get_num_threads())
    print("sklearn:", sklearn.__version__)
    print(
        "Metode: adversarial training (%s%s), eps=%s, eval_mode_adv=%s, robust_val_weight=%.2f, out=%s"
        % (
            args.adv_method.upper(),
            "-RS" if args.random_start else "",
            eps_choices,
            args.eval_mode_adv,
            args.robust_val_weight,
            args.out,
        )
    )

    train_ds = FrameDataset(os.path.join(args.data, "split_train.csv"), args.data, args.limit)
    val_ds = FrameDataset(os.path.join(args.data, "split_val.csv"), args.data, args.limit)
    test_ds = FrameDataset(os.path.join(args.data, "split_test.csv"), args.data, args.limit)
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        counts = Counter(ds.labels)
        print("%s: %d frame, kelas 0=%d, kelas 1=%d" % (name, len(ds), counts.get(0, 0), counts.get(1, 0)))
    test_counts = Counter(test_ds.labels)
    majority_acc = max(test_counts.values()) / len(test_ds)
    print("Akurasi tebakan kelas mayoritas di test: %.4f" % majority_acc)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameter model:", n_params)
    weight = None
    if not args.no_class_weight:
        counts = Counter(train_ds.labels)
        total = len(train_ds)
        if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
            print("PERINGATAN: satu kelas tidak ada di train (limit kecil?), class weight dimatikan untuk run ini.")
            weight = None
        else:
            weight = torch.tensor([total / (2.0 * counts[0]), total / (2.0 * counts[1])], dtype=torch.float32)
            print("Class weight aktif:", weight.tolist())
    else:
        print("Class weight NONAKTIF")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    log_rows = []
    best_score = -1.0
    best_epoch = -1
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch_adv(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            eps_choices=eps_choices,
            eval_mode_adv=args.eval_mode_adv,
            random_start=args.random_start,
            adv_method=args.adv_method,
            pgd_steps=args.pgd_steps,
            rng=rng,
            torch_gen=adv_gen,
        )
        val = evaluate(model, val_loader, criterion, device)
        if args.robust_val_weight > 0.0:
            val_adv = evaluate_adversarial(model, val_loader, criterion, device, val_eval_eps)
            val_adv_acc = val_adv["accuracy"]
            val_adv_macro_f1 = val_adv["macro_f1"]
            score = (1.0 - args.robust_val_weight) * val["macro_f1"] + args.robust_val_weight * val_adv_macro_f1
        else:
            val_adv_acc = None
            val_adv_macro_f1 = None
            score = val["macro_f1"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_adv_acc": "" if val_adv_acc is None else round(val_adv_acc, 6),
            "val_adv_macro_f1": "" if val_adv_macro_f1 is None else round(val_adv_macro_f1, 6),
            "val_score": round(score, 6),
        }
        log_rows.append(row)
        if val_adv_macro_f1 is not None:
            print(
                "Epoch %d/%d train_loss=%.4f train_acc=%.4f val_acc=%.4f val_f1=%.4f val_adv_f1=%.4f score=%.4f"
                % (epoch, args.epochs, train_loss, train_acc, val["accuracy"], val["macro_f1"], val_adv_macro_f1, score)
            )
        else:
            print(
                "Epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f"
                % (epoch, args.epochs, train_loss, train_acc, val["loss"], val["accuracy"], val["macro_f1"])
            )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": float(val["macro_f1"]),
                    "val_score": float(best_score),
                },
                os.path.join(args.out, "best_model.pt"),
            )
    torch.save({"epoch": args.epochs, "state_dict": model.state_dict()}, os.path.join(args.out, "final_model.pt"))
    elapsed = time.time() - started
    print("Training selesai dalam %.1f detik. Epoch terbaik: %d (score=%.4f)" % (elapsed, best_epoch, best_score))

    best = load_checkpoint_safe(os.path.join(args.out, "best_model.pt"), device)
    model.load_state_dict(best["state_dict"])
    test = evaluate(model, test_loader, criterion, device)
    print("TEST (model terbaik): acc=%.4f macro_f1=%.4f" % (test["accuracy"], test["macro_f1"]))
    print("  precision kelas 0=%.4f kelas 1=%.4f" % (test["precision"][0], test["precision"][1]))
    print("  recall kelas 0=%.4f kelas 1=%.4f" % (test["recall"][0], test["recall"][1]))
    print("  F1 kelas 0=%.4f kelas 1=%.4f" % (test["f1"][0], test["f1"][1]))
    print("  confusion matrix:", test["confusion_matrix"])

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_adv_acc",
        "val_adv_macro_f1",
        "val_score",
    ]
    with open(os.path.join(args.out, "training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    with open(os.path.join(args.out, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "majority_acc_test": majority_acc,
                "best_epoch": best_epoch,
                "version": "v2" if args.v2 else "v1",
                "train_config": {
                    "adv_method": args.adv_method,
                    "random_start": args.random_start,
                    "eval_mode_adv": args.eval_mode_adv,
                    "eps_choices": eps_choices,
                    "robust_val_weight": args.robust_val_weight,
                },
                "test": test,
            },
            f,
            indent=2,
        )
    save_confusion_matrix(test["confusion_matrix"], os.path.join(args.out, "confusion_matrix_test.png"))
    with open(os.path.join(args.out, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(
            "method=adversarial_training_%s random_start=%s eval_mode_adv=%s eps=%s robust_val_weight=%s data=%s epochs=%d batch_size=%d lr=%s seed=%d limit=%d class_weight=%s\n"
            % (
                args.adv_method,
                args.random_start,
                args.eval_mode_adv,
                ",".join(str(e) for e in eps_choices),
                args.robust_val_weight,
                args.data,
                args.epochs,
                args.batch_size,
                args.lr,
                args.seed,
                args.limit,
                not args.no_class_weight,
            )
        )
        f.write(
            "torch=%s sklearn=%s device=cpu params=%d elapsed_seconds=%.1f\n"
            % (torch.__version__, sklearn.__version__, n_params, elapsed)
        )
    print("Output tersimpan di:", args.out)


if __name__ == "__main__":
    main()
