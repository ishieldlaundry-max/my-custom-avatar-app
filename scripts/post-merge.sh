#!/bin/bash
set -e

# This project has heavyweight, hardware-specific Python dependencies and model
# downloads. Keep merge reconciliation non-destructive; environment setup is
# performed explicitly when a runnable ONNX or TensorRT target is chosen.
echo "No automatic post-merge setup is required."