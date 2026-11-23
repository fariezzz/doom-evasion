"""Record ViZDoom gameplay with synchronized CNN predictions and screenshots.

Run from the project directory:
    py scripts/demo_capture.py --model runs/baseline/best_model.pt --out runs/demo
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import vizdoom as vzd

try:
    from scripts.train_baseline import SmallCNN
except ImportError:
    from train_baseline import SmallCNN


CLASS_NAMES = ("TIDAK ADA MUSUH", "ADA MUSUH")


def to_gray_frame(buffer):
    if buffer.ndim == 3:
        if buffer.shape[0] == 1:
            return buffer[0]
        if buffer.shape[0] == 3:
            return cv2.cvtColor(np.transpose(buffer, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        return buffer[0]
    return buffer


def make_demo_frame(gray, step, true_label, prediction, confidence, scale):
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    view = cv2.resize(
        view,
        (gray.shape[1] * scale, gray.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    width = view.shape[1]
    header = np.full((42, width, 3), (24, 28, 35), dtype=np.uint8)
    footer = np.full((92, width, 3), (24, 28, 35), dtype=np.uint8)
    cv2.putText(header, "VIZDOOM  |  DEFEND THE CENTER", (18, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 220, 220), 2, cv2.LINE_AA)

    match = true_label == prediction
    status = "SESUAI" if match else "KELIRU"
    status_color = (100, 220, 130) if match else (80, 100, 255)
    cv2.putText(footer, "FRAME %04d   |   AI: %s (%.1f%%)" %
                (step, CLASS_NAMES[prediction], confidence * 100),
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(footer, "LABEL GAME: %s   |   HASIL: %s" %
                (CLASS_NAMES[true_label], status),
                (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
    return np.vstack((header, view, footer))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/baseline/best_model.pt")
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--capture-every", type=int, default=35)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--fps", type=float, default=35.0)
    args = parser.parse_args()

    if args.steps < 1 or args.capture_every < 1 or args.scale < 1 or args.fps <= 0:
        parser.error("--steps, --capture-every, --scale, and --fps must be positive")
    if any(os.path.exists(os.path.join(args.out, name)) for name in ("gameplay.avi", "gameplay.mp4")):
        parser.error("Output already exists; choose a new --out folder to avoid overwriting it")

    device = torch.device("cpu")
    model = SmallCNN()
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Model:", args.model, "| checkpoint epoch:", checkpoint.get("epoch"))

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_episode_timeout(args.steps)
    game.set_seed(args.seed)
    game.init()

    buttons = game.get_available_buttons()
    names = [str(button) for button in buttons]
    left = next((i for i, name in enumerate(names) if "TURN_LEFT" in name), None)
    right = next((i for i, name in enumerate(names) if "TURN_RIGHT" in name), None)
    attack = next((i for i, name in enumerate(names) if "ATTACK" in name), None)
    if left is None or right is None:
        game.close()
        raise RuntimeError("ViZDoom scenario is missing TURN_LEFT or TURN_RIGHT")

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "screenshots")
    os.makedirs(frames_dir, exist_ok=True)
    writer = None
    mp4_writer = None
    capture_rows = []
    rng = random.Random(args.seed)
    step = 0
    hold = 0
    move = None

    try:
        game.new_episode()
        while not game.is_episode_finished() and step < args.steps:
            state = game.get_state()
            if state is None:
                break

            gray = to_gray_frame(state.screen_buffer)
            labels = state.labels if state.labels is not None else []
            monster_boxes = [
                (int(label.x), int(label.y), int(label.width), int(label.height))
                for label in labels
                if getattr(label, "object_category", "") == "Monster"
            ]
            true_label = int(bool(monster_boxes))
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0]
            prediction = int(probabilities.argmax().item())
            confidence = float(probabilities[prediction].item())
            demo_frame = make_demo_frame(
                gray, step, true_label, prediction, confidence, args.scale
            )

            if writer is None:
                height, width = demo_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(
                    os.path.join(args.out, "gameplay.avi"), fourcc, args.fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not open video writer for gameplay.avi")
                mp4_writer = cv2.VideoWriter(
                    os.path.join(args.out, "gameplay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
                if not mp4_writer.isOpened():
                    raise RuntimeError("Could not open video writer for gameplay.mp4")
            writer.write(demo_frame)
            mp4_writer.write(demo_frame)

            if step % args.capture_every == 0:
                filename = "frame_%04d.png" % step
                screenshot_path = os.path.join(frames_dir, filename)
                if not cv2.imwrite(screenshot_path, demo_frame):
                    raise RuntimeError("Could not save screenshot: " + screenshot_path)
                capture_rows.append({
                    "video_frame": step,
                    "game_step": step,
                    "screenshot": os.path.join("screenshots", filename),
                    "label": true_label,
                    "prediction": prediction,
                    "confidence": "%.6f" % confidence,
                    "correct": int(true_label == prediction),
                })

            action = [0] * len(buttons)
            centered = attack is not None and any(
                abs((box[0] + box[2] / 2.0) - 160) <= 40 for box in monster_boxes
            )
            if centered:
                action[attack] = 1
            else:
                if hold <= 0:
                    move = rng.choice([left, right, None])
                    hold = rng.randint(5, 15)
                if move is not None:
                    action[move] = 1
                hold -= 1
            game.make_action(action, 1)
            step += 1
    finally:
        if writer is not None:
            writer.release()
        if mp4_writer is not None:
            mp4_writer.release()
        game.close()

    if not capture_rows:
        raise RuntimeError("ViZDoom produced no frames; no demo was saved")
    manifest_path = os.path.join(args.out, "capture_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        columns = ["video_frame", "game_step", "screenshot", "label",
                   "prediction", "confidence", "correct"]
        csv_writer = csv.DictWriter(file, fieldnames=columns)
        csv_writer.writeheader()
        csv_writer.writerows(capture_rows)

    print("Video:", os.path.join(args.out, "gameplay.avi"))
    print("Cuplikan layar:", frames_dir)
    print("Pencocokan frame/video:", manifest_path)
    print("Total frame video:", step, "| screenshot analisis:", len(capture_rows))


if __name__ == "__main__":
    main()
