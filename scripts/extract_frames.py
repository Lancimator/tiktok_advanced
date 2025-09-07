import sys
import os
import cv2


def extract_even_frames(input_path: str, output_dir: str, count: int = 12) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: iterate to count frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        total_frames = len(frames)
        if total_frames == 0:
            raise SystemExit("No frames found in video")
        indices = [int(i * (total_frames - 1) / max(1, count - 1)) for i in range(count)]
        for idx, i in enumerate(indices, 1):
            frame = frames[i]
            out_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
        return

    indices = [int(i * (total_frames - 1) / max(1, count - 1)) for i in range(count)]
    current = 0
    target_set = set(indices)
    saved = 0
    while current <= indices[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if current in target_set:
            saved += 1
            out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
        current += 1
    cap.release()


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/extract_frames.py <input_video> <output_dir> [count]", file=sys.stderr)
        raise SystemExit(2)
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    extract_even_frames(input_path, output_dir, count)


if __name__ == "__main__":
    main()

