"""Download the official MediaPipe Hand Landmarker model bundle."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official MediaPipe model to: {output}")
    urllib.request.urlretrieve(MODEL_URL, output)
    if output.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded model is unexpectedly small; download it again.")
    print(f"Done ({output.stat().st_size / 1024 / 1024:.1f} MiB).")


if __name__ == "__main__":
    main()
