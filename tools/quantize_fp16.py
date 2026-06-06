#!/usr/bin/env python3
"""Create an FP16-quantized TFLite model from a TensorFlow source model."""

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply TensorFlow Lite FP16 post-training quantization to a SavedModel "
            "directory or Keras model."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="TensorFlow SavedModel directory or .keras/.h5 model",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("facenet_fp16.tflite"),
        help="Output TFLite path (default: facenet_fp16.tflite)",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        parser.error(f"source model not found: {source}")
    if source.suffix.lower() == ".tflite":
        parser.error(
            "an existing .tflite file cannot be used as the quantization source; "
            "use the original SavedModel or Keras model"
        )
    if output.suffix.lower() != ".tflite":
        parser.error("output must use the .tflite extension")

    try:
        import tensorflow as tf
    except ImportError as error:
        parser.error(
            "TensorFlow is required on the development computer; "
            "install tools/requirements-model-compression.txt"
        )
        raise error

    if source.is_dir():
        converter = tf.lite.TFLiteConverter.from_saved_model(str(source))
    elif source.suffix.lower() in {".keras", ".h5"}:
        model = tf.keras.models.load_model(source, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        parser.error("source must be a SavedModel directory or .keras/.h5 model")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    quantized_model = converter.convert()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(quantized_model)

    size = output.stat().st_size
    print(f"Source:   {source}")
    print(f"Output:   {output}")
    print(f"FP16 size: {size} bytes ({size / 1024 / 1024:.2f} MiB)")
    print(f"SHA-256:  {sha256(output)}")
    print("The Android app still uses the original facenet.tflite model.")


if __name__ == "__main__":
    main()
