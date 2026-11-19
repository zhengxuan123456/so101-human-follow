"""Download the official MediaPipe BlazeFace short-range face detector."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "blaze_face_short_range.tflite",
    )
    output = parser.parse_args().output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official MediaPipe model to: {output}")
    urllib.request.urlretrieve(MODEL_URL, output)
    if output.stat().st_size < 100_000 or output.read_bytes()[4:8] != b"TFL3":
        raise RuntimeError("Downloaded face model is invalid; download it again.")
    print(f"Done ({output.stat().st_size / 1024:.0f} KiB).")


if __name__ == "__main__":
    main()
