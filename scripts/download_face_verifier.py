"""Download the OpenCV Zoo YuNet face verifier and its MIT license."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2026may.onnx"
)
LICENSE_URL = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE"
MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2026may.onnx",
    )
    output = parser.parse_args().output.resolve()
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        model = response.read()
    if len(model) < 200_000 or hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise RuntimeError("Downloaded YuNet model failed its size/SHA-256 check; no model was written.")
    with urllib.request.urlopen(LICENSE_URL, timeout=60) as response:
        license_text = response.read()
    if b"MIT License" not in license_text or b"Shiqi Yu" not in license_text:
        raise RuntimeError("The official YuNet license was not retrieved; no model was written.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(model)
    license_path = output.with_name(output.name + ".LICENSE")
    license_path.write_bytes(license_text)
    print(f"Saved verified YuNet model: {output} ({len(model) / 1024:.0f} KiB)")
    print(f"MIT license: {license_path}")


if __name__ == "__main__":
    main()
