"""Download the official MediaPipe Pose Landmarker Lite bundle."""
from pathlib import Path
import argparse
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['lite','full'],default='lite')
    args=parser.parse_args()
    path=ROOT/'models'/f'pose_landmarker_{args.variant}.task'
    url=MODEL_URL.replace('pose_landmarker_lite',f'pose_landmarker_{args.variant}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Model already present: {path}")
        return
    temporary = path.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=60) as response:
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if temporary.stat().st_size < 1_000_000:
        raise RuntimeError("Model download was unexpectedly small")
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Saved {path.name}; SHA256={digest}")


if __name__ == "__main__":
    main()
