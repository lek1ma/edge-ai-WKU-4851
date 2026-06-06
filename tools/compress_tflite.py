#!/usr/bin/env python3
"""Create a lossless Gzip copy of a TFLite model without modifying the source."""

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Losslessly compress a .tflite model into a separate .gz artifact."
    )
    parser.add_argument("model", type=Path, help="Input .tflite model")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: MODEL.gz)")
    parser.add_argument("--level", type=int, choices=range(1, 10), default=9)
    args = parser.parse_args()

    source = args.model.resolve()
    output = (args.output or Path(str(source) + ".gz")).resolve()
    if source.suffix.lower() != ".tflite":
        parser.error("input must be a .tflite file")
    if not source.is_file():
        parser.error(f"model not found: {source}")
    if source == output:
        parser.error("output must be different from the source model")

    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as raw, gzip.open(output, "wb", compresslevel=args.level) as packed:
        shutil.copyfileobj(raw, packed)

    original_size = source.stat().st_size
    compressed_size = output.stat().st_size
    reduction = 100.0 * (1.0 - compressed_size / original_size)
    print(f"Source:     {source}")
    print(f"Compressed: {output}")
    print(f"Size:       {original_size} -> {compressed_size} bytes ({reduction:.2f}% smaller)")
    print(f"SHA-256:    {sha256(source)}")
    print("The Android app still loads the original model; this artifact is not enabled at runtime.")


if __name__ == "__main__":
    main()
