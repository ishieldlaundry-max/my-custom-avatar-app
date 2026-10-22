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


@pytest.mark.parametrize("skin", [False, True])
def test_export_requires_actual_skin_binding(rig, tmp_path, skin):
    document = {
        "nodes": [{"mesh": 0, **({"skin": 0} if skin else {})}],
        "meshes": [{"primitives": [{"attributes": {"JOINTS_0": 0, "WEIGHTS_0": 1}}]}],
        "skins": [{"joints": [1]}] if skin else [],
    }
    body = json.dumps(document).encode()
    body += b" " * (-len(body) % 4)
    path = tmp_path / "mesh.glb"
    path.write_bytes(struct.pack("<4sIIII", b"glTF", 2, 20 + len(body),
                                 len(body), 0x4E4F534A) + body)
    if skin:
        rig.verify_export(path)
    else:
        with pytest.raises(RuntimeError, match="no usable skin"):
            rig.verify_export(path)
