---
name: GPU runtime limits
description: Hardware and package-index constraints affecting TensorRT validation in this workspace.
---

The hosted workspace may expose CUDA libraries bundled in Python wheels without exposing an NVIDIA kernel driver. In that state, `torch.cuda.is_available()` is false and the TensorRT 8 plugin cannot load because `libnvinfer.so.8` is unavailable. The Replit package index may also reject nested Nix CUDA package attributes even when the underlying nixpkgs channel contains them.

**Why:** Installing `pycuda` or a TensorRT Python package does not create the missing kernel-driver capability, and the TensorRT-specific `GridSample3D` path cannot run without the matching native runtime.

**How to apply:** Keep TensorRT selectable explicitly for a GPU deployment, but let automatic startup choose ONNX Runtime CPU when CUDA/TensorRT are not available. Validate that fallback with the standard ONNX warping model and the downloaded checkpoint tree.