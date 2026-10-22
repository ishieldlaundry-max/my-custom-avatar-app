"""Explicit NVIDIA inference policy; Windows never silently falls back to CPU."""
import os

_required = False


def nvidia_required():
    return _required or os.name == "nt"


def enforce_nvidia_runtime():
    global _required
    _required = True
    try:
        import torch
        if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to the main app's PyTorch environment.")
        name = torch.cuda.get_device_name(0)
        # Force a real CUDA allocation, not just device discovery.
        torch.empty(1, device="cuda:0")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "NVIDIA inference is required. The app will not fall back to Intel graphics "
            "or CPU inference. Check the NVIDIA driver and the main app's CUDA PyTorch "
            f"installation (not the separate Stable Fast 3D environment). Details: {exc}"
        ) from exc
    print(f"NVIDIA CUDA device: {name}", flush=True)
    return name


def onnx_session_policy(ort):
    options = ort.SessionOptions()
    available = ort.get_available_providers()
    if nvidia_required():
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "NVIDIA inference is required, but ONNX Runtime has no CUDA provider. "
                "The main app needs a compatible onnxruntime-gpu installation and CUDA/"
                "cuDNN runtime. CPU fallback is disabled. See WINDOWS_DESKTOP.md."
            )
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return [("CUDAExecutionProvider", {"device_id": 0})], options
    providers = [p for p in (
        "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"
    ) if p in available]
    if not providers:
        raise RuntimeError(f"No usable ONNX provider: {available}")
    return providers, options


def verify_onnx_session(session):
    if nvidia_required():
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA session initialization failed; CPU fallback is forbidden.")
        session.disable_fallback()
    print(f"Active ONNX providers: {session.get_providers()}", flush=True)
