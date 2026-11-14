"""Kumpul data final DoomEvasion di skenario defend_the_center.

Protokol final, disetujui user 14 November 2026 malam:
- Policy shoot: belok kiri/kanan dan diam, tembak kalau pusat kotak
  monster berada dalam pita tengah layar. Label ViZDoom hanya dipakai
  untuk menggerakkan kolektor, bukan input model.
- Lewati 25 langkah awal tiap episode (langkah 0-10 identik persis
  antar episode pada pilot, jadi harus dibuang supaya tidak bocor
  antar split).
- Simpan tiap 2 langkah (stride 2) untuk mengurangi frame berurutan
  yang identik.
- 30 episode, seed 400-429. Duplikat persis dihapus belakangan oleh
  build_dataset.py saat membangun split.
- Aturan label tetap: 1 kalau ada label kategori Monster, 0 kalau
  tidak. Kolom manifest sama dengan collect_data_center_v2.py.
Contoh:
    python collect_data_final.py --episodes 30 --max-steps 600 --seed-start 400 --out data_final --policy shoot --skip-steps 25 --stride 2
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import vizdoom as vzd


def to_gray84(buffer):
    if buffer.ndim == 3:
        if buffer.shape[0] == 1:
            gray = buffer[0]
        elif buffer.shape[0] == 3:
            gray = cv2.cvtColor(np.transpose(buffer, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        else:
            gray = buffer[0]
    else:
        gray = buffer
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--out", default="data_final")
    parser.add_argument("--policy", choices=["turn", "shoot"], default="shoot")
    parser.add_argument("--skip-steps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    print("Policy:", args.policy, "| skip-steps:", args.skip_steps, "| stride:", args.stride)

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_episode_timeout(args.max_steps)
    game.init()

    buttons = game.get_available_buttons()
    names = [str(b) for b in buttons]
    print("Tombol:", names)
    n_buttons = len(buttons)
    left = next((i for i, nm in enumerate(names) if "TURN_LEFT" in nm), None)
    right = next((i for i, nm in enumerate(names) if "TURN_RIGHT" in nm), None)
    attack = next((i for i, nm in enumerate(names) if "ATTACK" in nm), None)
    if left is None or right is None:
        raise SystemExit("Tombol TURN_LEFT/TURN_RIGHT tidak ditemukan: " + str(names))

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")
    total = {0: 0, 1: 0}
    first_shape = None
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "step", "path", "label", "label_names", "label_categories", "monster_count", "monster_boxes", "max_monster_h"])
        for ep in range(args.episodes):
            seed = args.seed_start + ep
            rng = random.Random(seed)
            game.set_seed(seed)
            game.new_episode()
            ep_dir = os.path.join(args.out, "ep%03d_seed%d" % (ep, seed))
            os.makedirs(ep_dir, exist_ok=True)
            counts = {0: 0, 1: 0}
            step = 0
            saved = 0
            hold = 0
            move = None
            while not game.is_episode_finished() and step < args.max_steps:
                state = game.get_state()
                if state is None:
                    break
                if first_shape is None:
                    first_shape = state.screen_buffer.shape
                    print("Shape buffer pertama:", first_shape)
                labels = state.labels if state.labels is not None else []
                label_names = [getattr(lb, "object_name", "?") for lb in labels]
                label_cats = [getattr(lb, "object_category", "?") for lb in labels]
                monster_boxes = []
                for lb in labels:
                    if getattr(lb, "object_category", "?") == "Monster":
                        monster_boxes.append((
                            int(getattr(lb, "x", 0)),
                            int(getattr(lb, "y", 0)),
                            int(getattr(lb, "width", 0)),
                            int(getattr(lb, "height", 0)),
                        ))
                max_monster_h = max([b[3] for b in monster_boxes], default=0)
                label = 1 if monster_boxes else 0
                should_save = step >= args.skip_steps and (step - args.skip_steps) % args.stride == 0
                if should_save:
                    frame = to_gray84(state.screen_buffer)
                    filename = "frame_%04d.png" % step
                    cv2.imwrite(os.path.join(ep_dir, filename), frame)
                    rel = os.path.basename(ep_dir) + "/" + filename
                    boxes_str = ";".join("%d,%d,%d,%d" % b for b in monster_boxes)
                    writer.writerow([ep, seed, step, rel, label, ";".join(label_names), ";".join(label_cats), len(monster_boxes), boxes_str, max_monster_h])
                    counts[label] += 1
                    total[label] += 1
                    saved += 1
                action = [0] * n_buttons
                centered = False
                if args.policy == "shoot" and attack is not None:
                    centered = any(abs((b[0] + b[2] / 2.0) - 160) <= 40 for b in monster_boxes)
                if centered:
                    action[attack] = 1
                else:
                    if hold <= 0:
                        move = rng.choice([left, right, None])
                        hold = rng.randint(5, 15)
                    if move is not None:
                        action[move] = 1
                    hold -= 1
                game.make_action(action)
                step += 1
            finished = game.is_episode_finished()
            print("Episode %d seed %d: %d langkah, %d frame tersimpan, kelas 0=%d, kelas 1=%d, episode_selesai=%s" % (ep, seed, step, saved, counts[0], counts[1], finished))
    game.close()
    print("Total tersimpan: %d frame, kelas 0=%d, kelas 1=%d" % (total[0] + total[1], total[0], total[1]))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
