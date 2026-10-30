"""Download the official MediaPipe Pose Landmarker Lite model bundle."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official MediaPipe model to: {output}")
    # Write directly to the destination. Some Windows data drives allow creating
    # files but deny atomic rename/delete, so a temporary-file rename is not portable.
    urllib.request.urlretrieve(MODEL_URL, output)
    if output.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded model is unexpectedly small; download it again.")
    print(f"Done ({output.stat().st_size / 1024 / 1024:.1f} MiB).")


if __name__ == "__main__":
    main()

