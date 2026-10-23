"""Create a local transparent PNG using rembg from the isolated SF3D environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from rembg import remove


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    with Image.open(source) as image:
        # Match OpenCV's orientation handling in the portrait renderer.
        rgba = remove(image.convert("RGBA"))
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(destination, format="PNG")


if __name__ == "__main__":
    main()
