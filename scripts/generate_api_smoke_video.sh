#!/bin/bash
set -e

SOURCE="${1:-assets/examples/driving/d0.mp4}"
OUTPUT="${2:-assets/examples/driving/d0-smoke.mp4}"

mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -hide_banner -loglevel error \
  -i "$SOURCE" \
  -frames:v 3 \
  -an \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -y "$OUTPUT"
echo "Wrote API smoke video to $OUTPUT"