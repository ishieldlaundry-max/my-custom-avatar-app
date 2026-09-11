import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
api = pytest.importorskip("api")


def readiness_payload(status):
    api.readiness_state.update(
        {
            "status": status,
            "backend": {"name": "onnx", "provider": "CPUExecutionProvider"},
            "checkpoints": {"status": "ready" if status == "ready" else status},
            "plugin": {
                "required": False,
                "status": "not_required" if status != "failed" else "failed",
            },
            "error": None if status != "failed" else "RuntimeError: unavailable",
        }
    )
    response = asyncio.run(api.health_ready())
    return response.status_code, json.loads(response.body)


def test_readiness_reports_starting():
    original = copy.deepcopy(api.readiness_state)
    try:
        assert api.backend_metadata("configs/onnx_infer.yaml") == {
            "name": "onnx",
            "provider": "CPUExecutionProvider",
            "plugin_required": False,
        }
        assert api.backend_metadata("configs/trt_infer.yaml") == {
            "name": "tensorrt",
            "provider": "TensorRTExecutionProvider",
            "plugin_required": True,
        }
        status_code, payload = readiness_payload("starting")
        assert status_code == 503
        assert payload["status"] == "starting"
        assert payload["backend"]["provider"] == "CPUExecutionProvider"
    finally:
        api.readiness_state.clear()
        api.readiness_state.update(original)


def test_readiness_reports_ready():
    original = copy.deepcopy(api.readiness_state)
    try:
        status_code, payload = readiness_payload("ready")
        assert status_code == 200
        assert payload["status"] == "ready"
        assert payload["checkpoints"]["status"] == "ready"
        assert payload["plugin"]["status"] == "not_required"
        assert payload["error"] is None
    finally:
        api.readiness_state.clear()
        api.readiness_state.update(original)


def test_readiness_reports_failed():
    original = copy.deepcopy(api.readiness_state)
    try:
        status_code, payload = readiness_payload("failed")
        assert status_code == 503
        assert payload["status"] == "failed"
        assert payload["error"].startswith("RuntimeError:")
    finally:
        api.readiness_state.clear()
        api.readiness_state.update(original)