import sys
import cv2


def main(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("error: cannot open video", file=sys.stderr)
        sys.exit(1)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0
    print(f"path: {path}")
    print(f"resolution: {width}x{height}")
    print(f"frames: {frames}")
    print(f"fps: {fps:.3f}")
    print(f"duration_sec: {duration:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/video_meta.py <video>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])

