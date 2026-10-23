"""Portable graph/numerical checks plus a real CUDA test when hardware exists."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

MODULE = Path(__file__).resolve().parents[1] / "src/models/cuda_warping.py"
spec = importlib.util.spec_from_file_location("cuda_warping", MODULE)
warping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warping)
MODEL = MODULE.parents[2] / "checkpoints/liveportrait_onnx/warping_spade.onnx"


def synthetic_model(padding="zeros", align=0):
    def value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 2, 2, 2])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2, 3])
    nodes = [
        helper.make_node("Add", ["source", "source"], ["early"]),
        helper.make_node("GridSample", ["early", "grid"], ["sample1"],
                         mode="bilinear", padding_mode=padding, align_corners=align),
        # Carries an early value across a sampling boundary.
        helper.make_node("Add", ["sample1", "early"], ["middle"]),
        helper.make_node("GridSample", ["middle", "grid"], ["sample2"],
                         mode="bilinear", padding_mode=padding, align_corners=align),
        helper.make_node("Add", ["sample2", "source"], ["out"]),
    ]
    model = helper.make_model(helper.make_graph(
        nodes, "test", [value("source"), grid], [value("out")],
        value_info=[value(n) for n in ("early", "sample1", "middle", "sample2")],
    ), opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 9
    return model


def cpu_reference(stages, inputs):
    """Test-only executor. The application never uses this CPU implementation."""
    values = dict(inputs)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    for kind, part in stages:
        if kind == "grid":
            values[part.output[0]] = warping.sample_grid(
                part, torch.from_numpy(values[part.input[0]]),
                torch.from_numpy(values[part.input[1]]),
            ).numpy()
        else:
            onnx.checker.check_model(part)
            session = ort.InferenceSession(part.SerializeToString(), options,
                                           providers=["CPUExecutionProvider"])
            outputs = session.run(None, {v.name: values[v.name] for v in session.get_inputs()})
            values.update(zip((v.name for v in session.get_outputs()), outputs))
    return values["out"]


class CudaWarpingTests(unittest.TestCase):
    def test_sampling_matches_onnx_across_boundaries(self):
        rng = np.random.default_rng(9)
        inputs = {"source": rng.normal(size=(1, 1, 2, 2, 2)).astype("float32"),
                  "grid": rng.uniform(-1.5, 1.5, (1, 2, 2, 2, 3)).astype("float32")}
        for padding in ("zeros", "border", "reflection"):
            for align in (0, 1):
                with self.subTest(padding=padding, align=align):
                    model = synthetic_model(padding, align)
                    expected = ort.InferenceSession(
                        model.SerializeToString(), providers=["CPUExecutionProvider"]
                    ).run(None, inputs)[0]
                    actual = cpu_reference(warping.split_grid_sample_graph(model), inputs)
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_partition_does_not_mutate_checkpoint(self):
        model = synthetic_model()
        before = model.SerializeToString()
        stages = warping.split_grid_sample_graph(model)
        self.assertEqual([kind for kind, _ in stages],
                         ["onnx", "grid", "onnx", "grid", "onnx"])
        self.assertEqual(before, model.SerializeToString())

    def test_missing_boundary_metadata_fails_explicitly(self):
        model = synthetic_model()
        del model.graph.value_info[:]
        with self.assertRaisesRegex(ValueError, "boundary shape"):
            warping.split_grid_sample_graph(model)

    def test_dynamic_shape_fails_explicitly(self):
        value = helper.make_tensor_value_info("dynamic", TensorProto.FLOAT, ["batch", 3])
        with self.assertRaisesRegex(ValueError, "static shapes"):
            warping.tensor_spec(value)

    @unittest.skipIf(torch.cuda.is_available(), "CPU host guard only")
    def test_no_silent_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "requires NVIDIA CUDA"):
            warping.CudaWarpingPredictor("unused.onnx")

    @unittest.skipUnless(MODEL.exists(), "Checkpoint not downloaded")
    def test_real_checkpoint_partitions(self):
        stages = warping.split_grid_sample_graph(onnx.load(MODEL))
        self.assertEqual(sum(kind == "grid" for kind, _ in stages), 2)
        for kind, part in stages:
            if kind == "onnx":
                onnx.checker.check_model(part)
                self.assertFalse(any(n.op_type == "GridSample" for n in part.graph.node))
                for value in (*part.graph.input, *part.graph.output):
                    warping.tensor_spec(value)

    @unittest.skipUnless(torch.cuda.is_available() and MODEL.exists(), "Requires NVIDIA and checkpoint")
    def test_real_cuda_render_matches_cpu_reference(self):
        rng = np.random.default_rng(13)
        inputs = {"feature_3d": rng.normal(0, .1, (1, 32, 16, 64, 64)).astype("float32"),
                  "kp_driving": rng.normal(0, .1, (1, 21, 3)).astype("float32"),
                  "kp_source": rng.normal(0, .1, (1, 21, 3)).astype("float32")}
        predictor = warping.CudaWarpingPredictor(str(MODEL))
        expected = cpu_reference(warping.split_grid_sample_graph(onnx.load(MODEL)), inputs)
        for _ in range(2):
            result = predictor.predict(*(inputs[name] for name, _, _ in predictor.inputs))[0]
            self.assertTrue(result.is_cuda)
            self.assertTrue(torch.isfinite(result).all().item())
            np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
