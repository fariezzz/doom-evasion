import argparse
import csv
import json
import os
import random
import time
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


class FrameDataset(Dataset):
    def __init__(self, csv_path, data_dir, limit=0):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if limit and limit > 0 and len(rows) > limit:
            step = max(1, len(rows) // limit)
            rows = rows[::step][:limit]
        self.items = []
        for r in rows:
            full = os.path.join(data_dir, r["path"].replace("/", os.sep))
            self.items.append((full, int(r["label"])))
        self.labels = [lb for _, lb in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        full, label = self.items[idx]
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Gagal membaca frame: " + full)
        arr = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, label


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            feat_size = self.features(torch.zeros(1, 1, 84, 84)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
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


def save_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0: tidak ada musuh", "1: ada musuh"])
    ax.set_yticklabels(["0: tidak ada musuh", "1: ada musuh"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label benar")
    ax.set_title("Confusion matrix test")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_final")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default="runs/baseline")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print("torch:", torch.__version__, "| device:", device, "| threads:", torch.get_num_threads())
    print("sklearn:", sklearn.__version__)

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
    best_macro_f1 = -1.0
    best_epoch = -1
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        })
        print("Epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f"
              % (epoch, args.epochs, train_loss, train_acc, val["loss"], val["accuracy"], val["macro_f1"]))
        if val["macro_f1"] > best_macro_f1:
            best_macro_f1 = val["macro_f1"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_macro_f1": best_macro_f1}, os.path.join(args.out, "best_model.pt"))
    torch.save({"epoch": args.epochs, "state_dict": model.state_dict()}, os.path.join(args.out, "final_model.pt"))
    elapsed = time.time() - started
    print("Training selesai dalam %.1f detik. Epoch terbaik: %d (val_macro_f1=%.4f)" % (elapsed, best_epoch, best_macro_f1))

    best = torch.load(os.path.join(args.out, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    test = evaluate(model, test_loader, criterion, device)
    print("TEST (model terbaik): acc=%.4f macro_f1=%.4f" % (test["accuracy"], test["macro_f1"]))
    print("  precision kelas 0=%.4f kelas 1=%.4f" % (test["precision"][0], test["precision"][1]))
    print("  recall kelas 0=%.4f kelas 1=%.4f" % (test["recall"][0], test["recall"][1]))
    print("  F1 kelas 0=%.4f kelas 1=%.4f" % (test["f1"][0], test["f1"][1]))
    print("  confusion matrix:", test["confusion_matrix"])
    print("  Gate 3: acc test %.4f vs mayoritas %.4f, F1 per kelas nol? %s"
          % (test["accuracy"], majority_acc, any(v == 0.0 for v in test["f1"])))

    with open(os.path.join(args.out, "training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(log_rows)
    with open(os.path.join(args.out, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"majority_acc_test": majority_acc, "best_epoch": best_epoch, "test": test}, f, indent=2)
    save_confusion_matrix(test["confusion_matrix"], os.path.join(args.out, "confusion_matrix_test.png"))
    with open(os.path.join(args.out, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write("data=%s epochs=%d batch_size=%d lr=%s seed=%d limit=%d class_weight=%s\n"
                % (args.data, args.epochs, args.batch_size, args.lr, args.seed, args.limit, not args.no_class_weight))
        f.write("torch=%s sklearn=%s device=cpu params=%d elapsed_seconds=%.1f\n"
                % (torch.__version__, sklearn.__version__, n_params, elapsed))
    print("Output tersimpan di:", args.out)


if __name__ == "__main__":
    main()
