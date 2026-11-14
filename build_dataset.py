"""Bangun dataset final DoomEvasion: hapus duplikat, lalu split per episode.

Membaca manifest.csv di folder data, menghapus frame duplikat persis
(hash piksel MD5, kemunculan pertama yang dipertahankan), lalu membagi
episode menjadi train/val/test 70/15/15 dengan shuffle deterministik.
Tidak ada episode yang sama di dua split, dan tiap split harus punya
dua kelas, kalau tidak script berhenti dengan pesan jelas.
Output di folder data yang sama: split_train.csv, split_val.csv,
split_test.csv (baris manifest yang lolos dedup), dan
dataset_summary.txt berisi hitungan lengkap.
Contoh:
    python build_dataset.py --data data_final --split-seed 12345
"""

import argparse
import csv
import hashlib
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_final")
    parser.add_argument("--split-seed", type=int, default=12345)
    args = parser.parse_args()

    manifest_path = os.path.join(args.data, "manifest.csv")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    raw_total = len(rows)
    print("Baris manifest mentah:", raw_total)

    seen = {}
    kept = []
    dropped = []
    for r in rows:
        img_path = os.path.join(args.data, r["path"].replace("/", os.sep))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit("Gagal membaca frame: " + img_path)
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen:
            dropped.append(r)
        else:
            seen[h] = r["path"]
            kept.append(r)
    print("Frame unik:", len(kept), "| duplikat dibuang:", len(dropped))

    episodes = sorted({(r["episode"], r["seed"]) for r in kept}, key=lambda t: int(t[1]))
    n = len(episodes)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    n_test = n - n_train - n_val
    order = list(episodes)
    random.Random(args.split_seed).shuffle(order)
    train_eps = set(order[:n_train])
    val_eps = set(order[n_train:n_train + n_val])
    test_eps = set(order[n_train + n_val:])
    assert not (train_eps & val_eps) and not (train_eps & test_eps) and not (val_eps & test_eps), "Episode bocor antar split"

    splits = {"train": [], "val": [], "test": []}
    for r in kept:
        key = (r["episode"], r["seed"])
        if key in train_eps:
            splits["train"].append(r)
        elif key in val_eps:
            splits["val"].append(r)
        elif key in test_eps:
            splits["test"].append(r)
        else:
            raise SystemExit("Episode tidak masuk split mana pun: " + str(key))
    for name, srows in splits.items():
        labels = Counter(r["label"] for r in srows)
        if set(labels) != {"0", "1"}:
            raise SystemExit("Split %s tidak punya dua kelas: %s" % (name, dict(labels)))

    for name, srows in splits.items():
        out_path = os.path.join(args.data, "split_%s.csv" % name)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(srows)
        print("Ditulis:", out_path, len(srows), "baris")

    kept_per_ep = Counter((r["episode"], r["seed"]) for r in kept)
    lines = []
    lines.append("Ringkasan dataset DoomEvasion")
    lines.append("Dibuat: " + datetime.now().isoformat(timespec="seconds"))
    lines.append("Folder data: " + args.data)
    lines.append("Split seed: %d" % args.split_seed)
    lines.append("Frame mentah: %d" % raw_total)
    lines.append("Frame unik setelah dedup: %d" % len(kept))
    lines.append("Duplikat dibuang: %d" % len(dropped))
    raw_labels = Counter(r["label"] for r in rows)
    kept_labels = Counter(r["label"] for r in kept)
    lines.append("Kelas mentah: 0=%d, 1=%d" % (raw_labels.get("0", 0), raw_labels.get("1", 0)))
    lines.append("Kelas unik: 0=%d, 1=%d" % (kept_labels.get("0", 0), kept_labels.get("1", 0)))
    lines.append("Jumlah episode: %d (train %d, val %d, test %d)" % (n, len(train_eps), len(val_eps), len(test_eps)))
    for name in ("train", "val", "test"):
        srows = splits[name]
        labels = Counter(r["label"] for r in srows)
        eps = sorted({(r["episode"], r["seed"]) for r in srows}, key=lambda t: int(t[1]))
        lines.append("Split %s: %d frame, kelas 0=%d, kelas 1=%d, episode: %s" % (name, len(srows), labels.get("0", 0), labels.get("1", 0), ",".join("ep%s_seed%s" % e for e in eps)))
    lines.append("Frame unik per episode:")
    for e in episodes:
        lines.append("  ep%s_seed%s: %d" % (e[0], e[1], kept_per_ep[e]))
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(args.data, "dataset_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
