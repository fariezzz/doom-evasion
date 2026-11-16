import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


class SmallMNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(path):
        print("Mengunduh MNIST dari %s ..." % MNIST_URL)
        try:
            urllib.request.urlretrieve(MNIST_URL, path)
        except urllib.error.URLError as exc:
            print("GAGAL mengunduh MNIST: %s" % exc)
            print("Periksa koneksi internet, lalu jalankan lagi.")
            raise SystemExit(1)
        print("Unduhan selesai: %d byte" % os.path.getsize(path))
    else:
        print("Memakai cache MNIST: %s" % path)
    arrays = np.load(path)
    return arrays["x_train"], arrays["y_train"], arrays["x_test"], arrays["y_test"]


def make_loaders(x_train, y_train, x_test, y_test, seed, batch_size, limit):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(x_train))
    val_idx, train_idx = order[:5000], order[5000:]

    def to_tensor(x, y):
        images = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
        labels = torch.from_numpy(y.astype(np.int64))
        return images, labels

    tr_x, tr_y = to_tensor(x_train[train_idx], y_train[train_idx])
    va_x, va_y = to_tensor(x_train[val_idx], y_train[val_idx])
    te_x, te_y = to_tensor(x_test, y_test)
    if limit > 0:
        tr_x, tr_y = tr_x[:limit], tr_y[:limit]
        va_x, va_y = va_x[:limit], va_y[:limit]
        te_x, te_y = te_x[:limit], te_y[:limit]

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True,
        num_workers=0, generator=gen,
    )
    val_loader = DataLoader(
        TensorDataset(va_x, va_y), batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(te_x, te_y), batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader, len(tr_x), len(va_x), len(te_x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        for true_label, pred_label in zip(labels.tolist(), pred.tolist()):
            confusion[true_label, pred_label] += 1
    return total_loss / total, correct / total, confusion


def save_confusion_matrix(confusion, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label benar")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_title("Confusion matrix test")
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_fgsm_examples(examples, epsilon, path):
    fig, axes = plt.subplots(len(examples), 3, figsize=(7, 2.4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, 0)
    for row, (clean, adv, true_label, clean_pred, adv_pred) in enumerate(examples):
        pert = np.clip((adv - clean) * 20.0 + 0.5, 0, 1)
        axes[row, 0].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("bersih")
        axes[row, 1].imshow(adv, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("adversarial eps=%s" % epsilon)
        axes[row, 2].imshow(pert, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("perturbasi x20")
        axes[row, 0].set_ylabel(
            "label %d: %d -> %d" % (true_label, clean_pred, adv_pred))
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fgsm_attack(model, loader, epsilon, device, max_examples=3):
    model.eval()
    initially_correct = 0
    flipped = 0
    adv_correct = 0
    total = 0
    examples = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad_(True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        with torch.no_grad():
            adv = torch.clamp(
                images + epsilon * images.grad.sign(), 0.0, 1.0)
            clean_pred = logits.argmax(dim=1)
            adv_pred = model(adv).argmax(dim=1)
            correct_mask = clean_pred == labels
            initially_correct += int(correct_mask.sum())
            adv_correct += int((adv_pred == labels).sum())
            total += len(labels)
            flipped_mask = correct_mask & (adv_pred != labels)
            flipped += int(flipped_mask.sum())
            if len(examples) < max_examples:
                for idx in torch.nonzero(flipped_mask).flatten().tolist():
                    if len(examples) >= max_examples:
                        break
                    examples.append((
                        images[idx, 0].detach().cpu().numpy(),
                        adv[idx, 0].detach().cpu().numpy(),
                        int(labels[idx]),
                        int(clean_pred[idx]),
                        int(adv_pred[idx]),
                    ))
    asr = flipped / initially_correct if initially_correct else 0.0
    return {
        "epsilon": epsilon,
        "asr": asr,
        "flipped": flipped,
        "denominator": initially_correct,
        "adv_accuracy": adv_correct / total if total else 0.0,
        "test_n": total,
        "examples_saved": len(examples),
    }, examples


def main():
    parser = argparse.ArgumentParser(description="Latihan MNIST: CNN + FGSM")
    parser.add_argument("--data", default="data_mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.01, 0.03, 0.05])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-fgsm", action="store_true")
    parser.add_argument("--out", default="runs/mnist_latihan")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print("torch: %s | device: %s | threads: %d"
          % (torch.__version__, device, torch.get_num_threads()))

    x_train, y_train, x_test, y_test = download_mnist(args.data)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = make_loaders(
        x_train, y_train, x_test, y_test, args.seed, args.batch_size, args.limit)
    print("train: %d gambar, val: %d gambar, test: %d gambar"
          % (n_train, n_val, n_test))

    model = SmallMNISTCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameter model: %d" % n_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "training_log.csv")
    best_val_acc = -1.0
    best_epoch = 0
    start = time.time()
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += len(labels)
            train_loss = total_loss / total
            train_acc = correct / total
            val_loss, val_acc, _ = evaluate(model, val_loader, device)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            print("Epoch %d/%d train_loss=%.4f train_acc=%.4f "
                  "val_loss=%.4f val_acc=%.4f"
                  % (epoch, args.epochs, train_loss, train_acc,
                     val_loss, val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_acc": val_acc},
                           os.path.join(args.out, "best_model.pt"))
    elapsed = time.time() - start
    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs,
                "val_acc": best_val_acc},
               os.path.join(args.out, "final_model.pt"))
    print("Training selesai dalam %.1f detik. Epoch terbaik: %d "
          "(val_acc=%.4f)" % (elapsed, best_epoch, best_val_acc))

    best = SmallMNISTCNN().to(device)
    checkpoint = torch.load(os.path.join(args.out, "best_model.pt"),
                            map_location=device, weights_only=True)
    best.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc, confusion = evaluate(best, test_loader, device)
    print("TEST (model terbaik): acc=%.4f" % test_acc)
    save_confusion_matrix(confusion,
                          os.path.join(args.out, "confusion_matrix_test.png"))
    with open(os.path.join(args.out, "test_metrics.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "val_acc": best_val_acc,
                   "test": {"loss": test_loss, "accuracy": test_acc,
                            "confusion_matrix": confusion.tolist(),
                            "n": n_test}},
                  handle, indent=1)

    with open(os.path.join(args.out, "run_config.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(
            "method=mnist_practice data=%s epochs=%d batch_size=%d lr=%s "
            "seed=%d limit=%d\n" % (args.data, args.epochs, args.batch_size,
                                    args.lr, args.seed, args.limit))
        handle.write("torch=%s device=cpu params=%d elapsed_seconds=%.1f\n"
                     % (torch.__version__, n_params, elapsed))

    if not args.no_fgsm:
        print("Akurasi bersih test: %.4f" % test_acc)
        results = []
        for epsilon in args.epsilons:
            result, examples = fgsm_attack(best, test_loader, epsilon, device)
            results.append(result)
            print("eps=%s ASR=%.4f (%d/%d) adv_acc=%.4f contoh=%d"
                  % (epsilon, result["asr"], result["flipped"],
                     result["denominator"], result["adv_accuracy"],
                     result["examples_saved"]))
            if examples:
                save_fgsm_examples(
                    examples, epsilon,
                    os.path.join(args.out, "examples_eps_%s.png"
                                 % str(epsilon).replace(".", "_")))
        with open(os.path.join(args.out, "fgsm_results.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"model": os.path.join(args.out, "best_model.pt"),
                       "data": args.data, "test_n": n_test,
                       "clean_accuracy": test_acc,
                       "per_epsilon": results}, handle, indent=1)

    print("Output tersimpan di: %s" % args.out)


if __name__ == "__main__":
    main()
