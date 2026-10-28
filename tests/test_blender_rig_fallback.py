"""CPU tests for the standalone Blender script; Blender itself is not imported."""
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


class Vector(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)

    @property
    def length_squared(self):
        return float(self.dot(self))


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", NS())
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    spec = importlib.util.spec_from_file_location(
        "rig_under_test", Path("scripts/rig_photo_mesh_blender.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_segment_distance_clamps_to_endpoints(rig):
    assert rig.segment_distance_squared(Vector([2, 1, 0]), Vector([0, 0, 0]),
                                        Vector([1, 0, 0])) == 2
    assert rig.segment_distance_squared(Vector([0, 1, 0]), Vector([0, 0, 0]),
                                        Vector([0, 0, 0])) == 1


@pytest.mark.parametrize("weights,valid", [
    ([], False), ([0], False), ([float("nan")], False), ([1], True),
])
def test_empty_groups_do_not_prove_weights(rig, weights, valid):
    mesh = NS(vertex_groups=[NS(index=0, name="Hips")],
              data=NS(vertices=[NS(groups=[NS(group=0, weight=w) for w in weights])]))
    armature = NS(data=NS(bones=[NS(name="Hips", use_deform=True)]))
    assert rig.has_complete_weights(mesh, armature) is valid


def test_fallback_normalizes_and_avoids_finger_weights(rig):
    class Groups(list):
        def new(self, name):
            group = NS(name=name, index=len(self))
            def add(indices, weight, mode):
                for index in indices:
                    vertices[index].groups.append(NS(group=group.index, weight=weight))
            group.add = add
            self.append(group)
            return group

    class Matrix:
        def __matmul__(self, other):
            return other
        def inverted(self):
            return self
        def copy(self):
            return self

    hips = NS(name="Hips", use_deform=True, parent=None,
              head_local=Vector([0, 0, 0]), tail_local=Vector([0, 0, 1]))
    spine = NS(name="Spine", use_deform=True, parent=hips,
               head_local=Vector([0, 0, 1]), tail_local=Vector([0, 0, 2]))
    finger = NS(name="LeftIndexProximal", use_deform=True)
    vertices = [NS(index=i, co=Vector([0.1, 0, z]), groups=[])
                for i, z in enumerate([0.2, 0.9, 1.5])]
    modifier = NS(type="ARMATURE")
    mesh = NS(vertex_groups=Groups(), data=NS(vertices=vertices),
              matrix_world=Matrix(), modifiers=[modifier])
    armature = NS(data=NS(bones=[hips, spine, finger]), matrix_world=Matrix())
    rig.proximity_weights(mesh, armature)
    assert rig.has_complete_weights(mesh, armature)
    assert [g.name for g in mesh.vertex_groups] == ["Hips", "Spine"]
    for vertex in vertices:
        assert sum(g.weight for g in vertex.groups) == pytest.approx(1)
    assert modifier.object is armature
    assert mesh.parent is armature


def _write_skinned_glb(
    path: Path,
    rig,
    *,
    include_skin: bool = True,
    missing_weight_joint: str | None = None,
    zero_weight_vertex: bool = False,
):
    joint_names = sorted(rig.REQUIRED_WEIGHTED_JOINTS)
    joint_indices = {name: index for index, name in enumerate(joint_names)}
    vertex_count = len(joint_names)
    joint_data = bytearray()
    weight_data = bytearray()
    for index, name in enumerate(joint_names):
        joint_index = index
        weights = (1.0, 0.0, 0.0, 0.0)
        if name == missing_weight_joint or (zero_weight_vertex and index == 0):
            joint_index = joint_indices["Hips"]
        if zero_weight_vertex and index == 0:
            weights = (0.0, 0.0, 0.0, 0.0)
        joint_data.extend(struct.pack("<4H", joint_index, 0, 0, 0))
        weight_data.extend(struct.pack("<4f", *weights))

    binary = bytes(joint_data + weight_data)
    joint_names_by_node = [
        {"name": name}
        for name in joint_names
    ]
    mesh_node = {"mesh": 0, **({"skin": 0} if include_skin else {})}
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(joint_data)},
            {
                "buffer": 0,
                "byteOffset": len(joint_data),
                "byteLength": len(weight_data),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5123,
                "count": vertex_count,
                "type": "VEC4",
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC4",
            },
        ],
        "nodes": [mesh_node, *joint_names_by_node],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "JOINTS_0": 0,
                            "WEIGHTS_0": 1,
                        }
                    }
                ]
            }
        ],
        "skins": [
            {"joints": list(range(1, len(joint_names) + 1))}
        ] if include_skin else [],
    }
    json_chunk = json.dumps(document).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\0" * (-len(binary) % 4)
    content = (
        struct.pack(
            "<4sII",
            b"glTF",
            2,
            12 + 8 + len(json_chunk) + 8 + len(binary),
        )
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    path.write_bytes(content)


@pytest.mark.parametrize("skin", [False, True])
def test_export_requires_actual_skin_binding(rig, tmp_path, skin):
    path = tmp_path / "mesh.glb"
    _write_skinned_glb(path, rig, include_skin=skin)
    if skin:
        rig.verify_export(path)
    else:
        with pytest.raises(RuntimeError, match="no usable skin"):
            rig.verify_export(path)


def test_export_rejects_required_body_joint_without_influencing_vertices(rig, tmp_path):
    path = tmp_path / "mesh.glb"
    _write_skinned_glb(path, rig, missing_weight_joint="Head")

    with pytest.raises(RuntimeError, match="no vertex weights.*Head"):
        rig.verify_export(path)


def test_export_rejects_vertices_without_any_positive_weight(rig, tmp_path):
    path = tmp_path / "mesh.glb"
    _write_skinned_glb(path, rig, zero_weight_vertex=True)

    with pytest.raises(RuntimeError, match="vertices with no usable skin weights"):
        rig.verify_export(path)
