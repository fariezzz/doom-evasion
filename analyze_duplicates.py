import argparse
import glob
import hashlib
import os
from collections import defaultdict

import cv2
import numpy as np


def load_frames(d):
    eps = {}
    for epdir in sorted(glob.glob(os.path.join(d, "ep*"))):
        name = os.path.basename(epdir)
        frames = {}
        for f in sorted(glob.glob(os.path.join(epdir, "frame_*.png"))):
            step = int(os.path.basename(f)[6:10])
            frames[step] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        eps[name] = frames
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    for d in args.dirs:
        print("===", d, "===")
        eps = load_frames(d)
        names = sorted(eps)
        for step in range(0, 31):
            base = eps[names[0]].get(step)
            if base is None:
                continue
            diffs = []
            for n in names[1:]:
                other = eps[n].get(step)
                if other is not None:
                    diffs.append(float(np.abs(base.astype(int) - other.astype(int)).mean()))
            if diffs:
                print("STEP %d mean_cross_diff=%.2f min=%.2f max=%.2f n=%d" % (step, float(np.mean(diffs)), min(diffs), max(diffs), len(diffs)))
        groups = defaultdict(list)
        for n in names:
            for step, img in eps[n].items():
                h = hashlib.md5(img.tobytes()).hexdigest()
                groups[h].append("%s/%04d" % (n, step))
        dup_groups = {h: v for h, v in groups.items() if len(v) > 1}
        dup_frames = sum(len(v) for v in dup_groups.values())
        total = sum(len(eps[n]) for n in names)
        print("TOTAL_FRAMES: %d DUPLICATE_FRAMES: %d DUP_GROUPS: %d" % (total, dup_frames, len(dup_groups)))
        for h, v in sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))[:10]:
            print("DUPGROUP n=%d %s" % (len(v), ",".join(v[:12])))


if __name__ == "__main__":
    main()
