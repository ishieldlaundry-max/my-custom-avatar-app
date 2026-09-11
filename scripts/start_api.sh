#!/bin/bash
source ~/.bashrc

# Optional: set FLIP_INFER_BACKEND=cpu, onnx, or trt before starting.
# GET /health/ready reports startup progress and whether /predict/ is usable.
python api.py