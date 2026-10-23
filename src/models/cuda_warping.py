"""Run volumetric GridSample in PyTorch CUDA, with the remaining graph in ORT.

No checkpoint conversion, native plugins, or CPU inference fallback is used.
"""
import copy

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F


def split_grid_sample_graph(model):
    """Partition a topologically sorted graph without losing cross-stage edges."""
    graph = model.graph
    values = {v.name: v for v in (*graph.input, *graph.value_info, *graph.output)}
    initializers = {v.name: v for v in graph.initializer}
    final_names = {v.name for v in graph.output}
    nodes = list(graph.node)
    stages = []
    start = 0

    def add_segment(end):
        segment = nodes[start:end]
        if not segment:
            return
        produced = {name for node in segment for name in node.output if name}
        consumed = list(dict.fromkeys(
            name for node in segment for name in node.input if name
        ))
        inputs = [name for name in consumed
                  if name not in produced and name not in initializers]
        later = {name for node in nodes[end:] for name in node.input} | final_names
        outputs = list(dict.fromkeys(
            name for node in segment for name in node.output if name in later
        ))
        missing = set(inputs + outputs) - values.keys()
        if missing:
            raise ValueError(f"Warping model lacks boundary shape metadata: {sorted(missing)}")
        subgraph = onnx.helper.make_graph(
            segment, f"cuda_warping_stage_{len(stages)}",
            [values[name] for name in inputs], [values[name] for name in outputs],
            [initializers[name] for name in consumed if name in initializers],
        )
        submodel = copy.deepcopy(model)
        submodel.graph.CopyFrom(subgraph)
        stages.append(("onnx", submodel))

    for index, node in enumerate(nodes):
        if node.op_type != "GridSample":
            continue
        if node.domain not in ("", "ai.onnx"):
            raise ValueError(f"Unsupported GridSample domain: {node.domain}")
        add_segment(index)
        stages.append(("grid", node))
        start = index + 1
    add_segment(len(nodes))
    if not any(kind == "grid" for kind, _ in stages):
        raise ValueError("Expected GridSample nodes in the warping model")
    return stages


def sample_grid(node, source, grid):
    """Preserve the ONNX node's sampling semantics, including align_corners."""
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    mode = attrs.get("mode", b"bilinear").decode()
    mode = {"linear": "bilinear", "cubic": "bicubic"}.get(mode, mode)
    padding = attrs.get("padding_mode", b"zeros").decode()
    return F.grid_sample(source, grid, mode=mode, padding_mode=padding,
                         align_corners=bool(attrs.get("align_corners", 0)))


def tensor_spec(value):
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))
    dims = value.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") for dim in dims):
        raise ValueError(f"CUDA warping requires static shapes: {value.name}")
    return value.name, tuple(dim.dim_value for dim in dims), dtype


class CudaWarpingPredictor:
    """CUDA-only inference with explicit I/O binding across ORT/PyTorch stages."""

    def __init__(self, model_path, **kwargs):
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            raise RuntimeError("The warping compatibility path requires NVIDIA CUDA.")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("The warping compatibility path requires ONNX Runtime CUDA.")
        self.device = torch.device("cuda:0")
        model = onnx.load(model_path)
        self.inputs = [tensor_spec(v) for v in model.graph.input]
        self.outputs = [tensor_spec(v) for v in model.graph.output]
        self.stages = []
        for kind, part in split_grid_sample_graph(model):
            if kind == "grid":
                self.stages.append((kind, part))
                continue
            opts = ort.SessionOptions()
            opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(
                part.SerializeToString(), sess_options=opts,
                providers=[("CUDAExecutionProvider", {"device_id": 0})],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("Warping stage failed to initialize CUDA; CPU fallback forbidden.")
            session.disable_fallback()
            self.stages.append((kind, (session, [tensor_spec(v) for v in part.graph.output])))
        print("Warping backend: ONNX CUDA + PyTorch CUDA GridSample (CPU fallback disabled)",
              flush=True)

    def input_spec(self):
        return self.inputs

    def output_spec(self):
        return self.outputs

    @torch.inference_mode()
    def predict(self, *data):
        if len(data) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} warping inputs, got {len(data)}")
        tensors = {}
        for (name, shape, dtype), value in zip(self.inputs, data):
            tensor = torch.as_tensor(value, device=self.device,
                                     dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype).contiguous()
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Invalid warping input {name}: {tuple(tensor.shape)} != {shape}")
            tensors[name] = tensor
        for index, (kind, stage) in enumerate(self.stages):
            if kind == "grid":
                tensors[stage.output[0]] = sample_grid(
                    stage, tensors[stage.input[0]], tensors[stage.input[1]]
                ).contiguous()
            else:
                session, outputs = stage
                binding = session.io_binding()
                for value in session.get_inputs():
                    tensor = tensors[value.name]
                    dtype = np.dtype(torch.empty((), dtype=tensor.dtype).numpy().dtype)
                    binding.bind_input(value.name, "cuda", 0, dtype,
                                       tuple(tensor.shape), tensor.data_ptr())
                for name, shape, dtype in outputs:
                    tensor = torch.empty(shape, device=self.device,
                                         dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype)
                    tensors[name] = tensor
                    binding.bind_output(name, "cuda", 0, dtype, shape, tensor.data_ptr())
                # ORT and PyTorch own distinct streams. Synchronize explicitly before
                # handing off pointers, and before releasing any input allocations.
                torch.cuda.synchronize(self.device)
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                torch.cuda.synchronize(self.device)
            # Release large dense-motion intermediates as soon as they are unused.
            needed = {name for name, _, _ in self.outputs}
            for later_kind, later in self.stages[index + 1:]:
                needed.update(later.input if later_kind == "grid"
                              else (value.name for value in later[0].get_inputs()))
            tensors = {name: value for name, value in tensors.items() if name in needed}
        return [tensors[name] for name, _, _ in self.outputs]
