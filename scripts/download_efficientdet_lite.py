#!/usr/bin/env python3
"""Download the EfficientDet-Lite TFLite model used by the Android app.

I originally pulled the model with:

curl -L -o /tmp/efficientdet_lite0_detection.tflite \
  'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/default/1?lite-format=tflite'

This script makes that step repeatable.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODEL_URLS = {
    "lite0": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/default/1?lite-format=tflite",
    "lite1": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/default/1?lite-format=tflite",
    "lite2": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/default/1?lite-format=tflite",
    "lite3": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite3/detection/default/1?lite-format=tflite",
    "lite4": "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite4/detection/default/1?lite-format=tflite",
}

DEFAULT_OUTPUT = Path("app/src/main/assets/efficientdet_lite0_detection.tflite")


def download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as output:
            while chunk := response.read(1 << 20):
                output.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done // total
                    print(f"\r  {done >> 20} / {total >> 20} MiB ({pct}%)", end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=MODEL_URLS.keys(),
        default="lite0",
        help="EfficientDet-Lite variant to download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .tflite path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    print(f"Downloading EfficientDet-{args.variant} detection TFLite")
    print(f"URL: {MODEL_URLS[args.variant]}")
    download_with_progress(MODEL_URLS[args.variant], args.output)
    mb = args.output.stat().st_size / 1_000_000
    print(f"Wrote {args.output} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
