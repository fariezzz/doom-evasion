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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--out", default="data_raw")
    args = parser.parse_args()

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_episode_timeout(args.max_steps)
    game.init()

    buttons = game.get_available_buttons()
    names = [str(b) for b in buttons]
    print("Tombol:", names)
    n_buttons = len(buttons)
    left = next((i for i, nm in enumerate(names) if "MOVE_LEFT" in nm), None)
    right = next((i for i, nm in enumerate(names) if "MOVE_RIGHT" in nm), None)
    if left is None or right is None:
        raise SystemExit("Tombol MOVE_LEFT/MOVE_RIGHT tidak ditemukan: " + str(names))

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")
    total = {0: 0, 1: 0}
    first_shape = None
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "step", "path", "label", "label_names", "label_categories"])
        for ep in range(args.episodes):
            seed = args.seed_start + ep
            rng = random.Random(seed)
            game.set_seed(seed)
            game.new_episode()
            ep_dir = os.path.join(args.out, "ep%03d_seed%d" % (ep, seed))
            os.makedirs(ep_dir, exist_ok=True)
            counts = {0: 0, 1: 0}
            step = 0
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
                label = 1 if "Monster" in label_cats else 0
                frame = to_gray84(state.screen_buffer)
                filename = "frame_%04d.png" % step
                cv2.imwrite(os.path.join(ep_dir, filename), frame)
                rel = os.path.basename(ep_dir) + "/" + filename
                writer.writerow([ep, seed, step, rel, label, ";".join(label_names), ";".join(label_cats)])
                counts[label] += 1
                total[label] += 1
                if hold <= 0:
                    move = rng.choice([left, right, None])
                    hold = rng.randint(5, 15)
                action = [0] * n_buttons
                if move is not None:
                    action[move] = 1
                hold -= 1
                game.make_action(action)
                step += 1
            print("Episode %d seed %d: %d frame, kelas 0=%d, kelas 1=%d" % (ep, seed, step, counts[0], counts[1]))
    game.close()
    print("Total: %d frame, kelas 0=%d, kelas 1=%d" % (total[0] + total[1], total[0], total[1]))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
