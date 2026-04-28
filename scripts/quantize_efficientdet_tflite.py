#!/usr/bin/env python3
"""Convert an EfficientDet SavedModel to TFLite, with optional full INT8.

The app's checked-in EfficientDet-Lite0 model is already quantized and uses
byte input. Use this script when you have a SavedModel export and want to
produce a replacement TFLite file. If --saved-model-dir is omitted, the script
downloads the TensorFlow/Kaggle EfficientDet-Lite SavedModel source.

Usage
-----
# Full INT8 for NPU testing, using the Kaggle/TFHub source and COCO calibration:
python scripts/quantize_efficientdet_tflite.py --full-int8

# Full INT8 from a local SavedModel:
python scripts/quantize_efficientdet_tflite.py --saved-model-dir build/efficientdet_saved_model --full-int8

# Full INT8 with your own JPEG calibration set and copy into app assets:
python scripts/quantize_efficientdet_tflite.py --full-int8 --calib-dir /path/to/jpegs --copy-to-assets
"""

from __future__ import annotations

import argparse
import collections
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf


DEFAULT_OUTPUT_DIR = Path("build/model_exports/efficientdet_lite")
DEFAULT_SOURCE_DIR = Path("build/model_sources/efficientdet_lite")
CALIB_CACHE_DIR = Path("build/calib_images")

COCO_ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_ENTRY = "annotations/instances_val2017.json"
COCO_IMG_BASE = "http://images.cocodataset.org/val2017"

KAGGLE_HANDLES = {
    "lite0": "tensorflow/efficientdet/tensorFlow2/lite0-detection/1",
    "lite1": "tensorflow/efficientdet/tensorFlow2/lite1-detection/1",
    "lite2": "tensorflow/efficientdet/tensorFlow2/lite2-detection/1",
    "lite3": "tensorflow/efficientdet/tensorFlow2/lite3-detection/1",
    "lite4": "tensorflow/efficientdet/tensorFlow2/lite4-detection/1",
}

APP_MODEL_ASSET = Path("app/src/main/assets/efficientdet_lite0_detection.tflite")


def download_with_progress(url: str, dest: Path) -> None:
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


def fetch_annotations(dest_json: Path) -> None:
    print("Downloading COCO val2017 annotations (one-time, about 241 MB)")
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        tmp = Path(temp_file.name)
    try:
        download_with_progress(COCO_ANN_ZIP_URL, tmp)
        with zipfile.ZipFile(tmp) as zip_file:
            with zip_file.open(COCO_ANN_ENTRY) as source:
                dest_json.write_bytes(source.read())
        print(f"Saved image list to {dest_json}")
    finally:
        tmp.unlink(missing_ok=True)


def ensure_coco_images(dest: Path, count: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ann_path = dest / "instances_val2017.json"
    if not ann_path.exists():
        fetch_annotations(ann_path)

    with ann_path.open() as file:
        images = json.load(file)["images"]

    filenames = [image["file_name"] for image in images[:count]]
    needed = [name for name in filenames if not (dest / name).exists()]
    if not needed:
        print(f"Using {len(filenames)} cached COCO calibration images from {dest}")
        return

    print(f"Downloading {len(needed)} COCO val2017 calibration images into {dest}")
    for index, filename in enumerate(needed, 1):
        urllib.request.urlretrieve(f"{COCO_IMG_BASE}/{filename}", dest / filename)
        print(f"\r  {index}/{len(needed)}: {filename}", end="", flush=True)
    print()


def make_representative_dataset(calib_dir: Path, input_size: int, num_images: int):
    try:
        import cv2  # type: ignore[import-untyped]

        def load_image(path: Path) -> np.ndarray:
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(image, (input_size, input_size))

    except ImportError:
        from PIL import Image

        def load_image(path: Path) -> np.ndarray:
            image = Image.open(path).convert("RGB").resize((input_size, input_size))
            return np.array(image)

    images = sorted(calib_dir.glob("*.jpg"))[:num_images]
    if not images:
        raise FileNotFoundError(f"No JPEG images found in {calib_dir}")
    print(f"Calibrating with {len(images)} images from {calib_dir}")

    def generator():
        for path in images:
            # The Kaggle/TFHub EfficientDet-Lite detection SavedModels expect
            # uint8 RGB [1,H,W,3] in the 0..255 range.
            tensor = load_image(path).astype(np.uint8)[np.newaxis]
            yield [tensor]

    return generator


def download_saved_model(variant: str, output_dir: Path, force: bool) -> Path:
    handle = KAGGLE_HANDLES[variant]
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to auto-download the EfficientDet SavedModel. "
            "Install it with: python3 -m pip install kagglehub"
        ) from exc

    print(f"Downloading EfficientDet-{variant} SavedModel from Kaggle: {handle}")
    path = Path(
        kagglehub.model_download(
            handle,
            output_dir=str(output_dir),
            force_download=force,
        ),
    )
    if not (path / "saved_model.pb").exists():
        raise FileNotFoundError(f"Kaggle download did not produce a SavedModel at {path}")
    return path


def convert(
    saved_model_dir: Path,
    output_dir: Path,
    variant: str,
    input_size: int,
    full_int8: bool,
    calib_dir: Path | None,
    num_calib_images: int,
    float_io: bool,
    allow_select_tf_ops: bool,
) -> Path:
    if not saved_model_dir.is_dir():
        raise FileNotFoundError(f"SavedModel not found: {saved_model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_model = tf.saved_model.load(str(saved_model_dir))
    signature = saved_model.signatures["serving_default"]

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, input_size, input_size, 3], tf.uint8, name="images"),
        ],
    )
    def fixed_shape_detector(images):
        outputs = signature(images=images)
        return [
            tf.cast(outputs["output_0"], tf.float32),  # boxes, [1,N,4], input-pixel coords
            tf.cast(outputs["output_2"], tf.float32),  # classes, [1,N]
            tf.cast(outputs["output_1"], tf.float32),  # scores, [1,N]
            tf.cast(outputs["output_3"], tf.float32),  # count, [1]
        ]

    concrete_func = fixed_shape_detector.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if full_int8:
        if calib_dir is None:
            ensure_coco_images(CALIB_CACHE_DIR, num_calib_images)
            calib_dir = CALIB_CACHE_DIR
        converter.representative_dataset = make_representative_dataset(
            calib_dir,
            input_size,
            num_calib_images,
        )
        if allow_select_tf_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if float_io:
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            suffix = "full_int8_floatio"
        else:
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
            suffix = "full_int8_uint8in_floatout"
    else:
        suffix = "dynamic_range"

    print(f"Converting EfficientDet SavedModel ({suffix})")
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        if full_int8 and not allow_select_tf_ops:
            raise RuntimeError(
                "Pure built-in INT8 conversion failed. The Kaggle SavedModel contains "
                "TensorList ops in its dynamic preprocessing/postprocessing path. For a "
                "convertible-but-not-NPU-clean fallback, rerun with --allow-select-tf-ops. "
                "For Qualcomm NPU testing, prefer the pre-exported EfficientDet-Lite TFLite "
                "asset downloaded by scripts/download_efficientdet_lite.py; it is already "
                "mostly int8/uint8 and avoids Flex ops."
            ) from exc
        raise
    output = output_dir / f"efficientdet_{variant}_{input_size}_{suffix}.tflite"
    output.write_bytes(tflite_model)
    mb = output.stat().st_size / 1_000_000
    print(f"Wrote {output} ({mb:.1f} MB)")
    print_tflite_summary(output)
    if allow_select_tf_ops:
        print(
            "Warning: this model contains Select TF/Flex ops. It is useful as a "
            "conversion fallback, but it is not the artifact to benchmark on Qualcomm NPU.",
        )
    return output


def print_tflite_summary(model_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    dtype_counts = collections.Counter(str(detail["dtype"]) for detail in interpreter.get_tensor_details())
    print("Tensor dtype counts:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count}")
    print("Inputs:")
    for detail in interpreter.get_input_details():
        print(f"  {detail['name']} shape={detail['shape'].tolist()} dtype={detail['dtype']} quant={detail['quantization']}")
    print("Outputs:")
    for detail in interpreter.get_output_details():
        print(f"  {detail['name']} shape={detail['shape'].tolist()} dtype={detail['dtype']} quant={detail['quantization']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--saved-model-dir",
        type=Path,
        default=None,
        help="Local EfficientDet SavedModel. If omitted, downloads from Kaggle/TFHub.",
    )
    parser.add_argument("--variant", choices=KAGGLE_HANDLES.keys(), default="lite0")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=320, choices=[320, 384, 448, 512, 640])
    parser.add_argument("--full-int8", action="store_true", help="Full W8A8 INT8 quantization for NPU testing.")
    parser.add_argument(
        "--float-io",
        action="store_true",
        help="Use float input/output for full INT8 models. The app currently expects uint8 input.",
    )
    parser.add_argument(
        "--allow-select-tf-ops",
        action="store_true",
        help="Allow Flex/Select TF ops if pure TFLite INT8 conversion fails. This is not NPU-clean.",
    )
    parser.add_argument("--calib-dir", type=Path, default=None, help="Directory of JPEG calibration images.")
    parser.add_argument("--num-calib-images", type=int, default=200)
    parser.add_argument(
        "--copy-to-assets",
        action="store_true",
        help=f"Copy the converted model to {APP_MODEL_ASSET}.",
    )
    args = parser.parse_args()

    saved_model_dir = args.saved_model_dir or download_saved_model(
        args.variant,
        args.source_dir / args.variant,
        args.force_download,
    )
    output = convert(
        saved_model_dir=saved_model_dir,
        output_dir=args.output_dir,
        variant=args.variant,
        input_size=args.input_size,
        full_int8=args.full_int8,
        calib_dir=args.calib_dir,
        num_calib_images=args.num_calib_images,
        float_io=args.float_io,
        allow_select_tf_ops=args.allow_select_tf_ops,
    )
    if args.copy_to_assets:
        APP_MODEL_ASSET.parent.mkdir(parents=True, exist_ok=True)
        APP_MODEL_ASSET.write_bytes(output.read_bytes())
        print(f"Copied {output} to {APP_MODEL_ASSET}")


if __name__ == "__main__":
    main()
