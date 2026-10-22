from types import SimpleNamespace as NS

import pytest

from src.utils import nvidia_runtime as runtime
import desktop_launcher


class Options:
    def __init__(self):
        self.entries = {}
    def add_session_config_entry(self, name, value):
        self.entries[name] = value


@pytest.mark.parametrize("available", [
    ["CPUExecutionProvider"], ["DmlExecutionProvider", "CPUExecutionProvider"], [],
])
def test_strict_policy_refuses_cpu_and_intel(monkeypatch, available):
    monkeypatch.setattr(runtime, "nvidia_required", lambda: True)
    ort = NS(SessionOptions=Options, get_available_providers=lambda: available)
    with pytest.raises(RuntimeError, match="no CUDA provider"):
        runtime.onnx_session_policy(ort)


def test_strict_policy_disables_cpu_nodes(monkeypatch):
    monkeypatch.setattr(runtime, "nvidia_required", lambda: True)
    ort = NS(SessionOptions=Options, get_available_providers=lambda: [
        "CUDAExecutionProvider", "CPUExecutionProvider"])
    providers, options = runtime.onnx_session_policy(ort)
    assert providers == [("CUDAExecutionProvider", {"device_id": 0})]
    assert options.entries["session.disable_cpu_ep_fallback"] == "1"


def test_failed_cuda_session_is_rejected(monkeypatch):
    monkeypatch.setattr(runtime, "nvidia_required", lambda: True)
    with pytest.raises(RuntimeError, match="forbidden"):
        runtime.verify_onnx_session(NS(get_providers=lambda: ["CPUExecutionProvider"]))


def test_successful_cuda_disables_runtime_fallback(monkeypatch):
    monkeypatch.setattr(runtime, "nvidia_required", lambda: True)
    calls = []
    session = NS(get_providers=lambda: ["CUDAExecutionProvider"],
                 disable_fallback=lambda: calls.append(True))
    runtime.verify_onnx_session(session)
    assert calls == [True]


def test_linux_preview_can_use_cpu(monkeypatch):
    monkeypatch.setattr(runtime, "nvidia_required", lambda: False)
    providers, options = runtime.onnx_session_policy(
        NS(SessionOptions=Options, get_available_providers=lambda: ["CPUExecutionProvider"]))
    assert providers == ["CPUExecutionProvider"]
    assert not options.entries


def test_cuda_preflight_checks_allocation(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(runtime, "_required", False)
    monkeypatch.setitem(sys.modules, "torch", NS(
        version=NS(hip=None), cuda=NS(is_available=lambda: True,
                                    get_device_name=lambda i: "NVIDIA test"),
        empty=lambda size, device: calls.append(device)))
    assert runtime.enforce_nvidia_runtime() == "NVIDIA test"
    assert calls == ["cuda:0"]


def test_launcher_does_not_use_sf3d_python(tmp_path):
    with pytest.raises(RuntimeError, match="existing venv"):
        desktop_launcher.app_python(tmp_path)
    (tmp_path / "venv/Scripts").mkdir(parents=True)
    python = tmp_path / "venv/Scripts/python.exe"
    python.touch()
    (tmp_path / "webui.py").touch()
    assert desktop_launcher.app_python(tmp_path) == python
