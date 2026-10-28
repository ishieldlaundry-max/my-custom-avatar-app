"""Bind a full-body GLB to a simple humanoid armature using Blender's local API."""

import sys
import json
import math
import struct
from pathlib import Path

import bpy
from mathutils import Vector


REQUIRED_WEIGHTED_JOINTS = {
    "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
}


def fail(message):
    print(f"RIGGING_ERROR: {message}", file=sys.stderr)
    raise RuntimeError(message)


def parse_arguments():
    if "--" not in sys.argv:
        fail("Expected input and output paths after --.")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        fail("Expected exactly an input GLB and output GLB path.")
    return Path(args[0]).resolve(), Path(args[1]).resolve()


def mesh_bounds(meshes):
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in meshes
        for corner in obj.bound_box
    ]
    if not points:
        fail("The input GLB contains no mesh geometry.")
    return (
        min(point.x for point in points),
        max(point.x for point in points),
        min(point.y for point in points),
        max(point.y for point in points),
        min(point.z for point in points),
        max(point.z for point in points),
    )


def add_bone(edit_bones, name, head, tail, parent=None):
    bone = edit_bones.new(name)
    bone.head = head
    bone.tail = tail
    bone.parent = parent
    return bone


def create_humanoid_armature(bounds):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    width = max(max_x - min_x, (max_z - min_z) * 0.25, 0.01)
    height = max(max_z - min_z, 0.01)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    at = lambda height_ratio: min_z + height * height_ratio
    armature_data = bpy.data.armatures.new("LocalPhotoHumanoid")
    armature = bpy.data.objects.new("LocalPhotoHumanoid", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = armature_data.edit_bones

    hips = add_bone(bones, "Hips", (center_x, center_y, at(0.43)), (center_x, center_y, at(0.55)))
    spine = add_bone(bones, "Spine", hips.tail, (center_x, center_y, at(0.67)), hips)
    chest = add_bone(bones, "Chest", spine.tail, (center_x, center_y, at(0.78)), spine)
    neck = add_bone(bones, "Neck", chest.tail, (center_x, center_y, at(0.86)), chest)
    add_bone(bones, "Head", neck.tail, (center_x, center_y, at(0.99)), neck)

    for side, sign in (("Left", 1), ("Right", -1)):
        shoulder_x = center_x + sign * width * 0.17
        elbow_x = center_x + sign * width * 0.29
        wrist_x = center_x + sign * width * 0.39
        hand_x = center_x + sign * width * 0.46
        upper_arm = add_bone(
            bones, f"{side}UpperArm",
            (shoulder_x, center_y, at(0.77)),
            (elbow_x, center_y, at(0.68)),
            chest,
        )
        lower_arm = add_bone(
            bones, f"{side}LowerArm",
            upper_arm.tail,
            (wrist_x, center_y, at(0.60)),
            upper_arm,
        )
        hand = add_bone(
            bones, f"{side}Hand",
            lower_arm.tail,
            (hand_x, center_y, at(0.58)),
            lower_arm,
        )
        for finger_index, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Little")):
            finger_start_x = hand_x - sign * width * (0.018 + finger_index * 0.002)
            finger_y = center_y + height * (finger_index - 2) * 0.002
            finger_z = at(0.58)
            if finger == "Thumb":
                proximal = add_bone(
                    bones, f"{side}ThumbProximal",
                    (finger_start_x, finger_y, finger_z),
                    (finger_start_x - sign * width * 0.012, finger_y, finger_z - height * 0.018),
                    hand,
                )
                add_bone(
                    bones, f"{side}ThumbDistal",
                    proximal.tail,
                    (proximal.tail.x - sign * width * 0.009, proximal.tail.y, proximal.tail.z - height * 0.014),
                    proximal,
                )
                continue
            previous = hand
            start = (finger_start_x, finger_y, finger_z)
            for joint_index, joint_name in enumerate(("Proximal", "Intermediate", "Distal")):
                length = height * (0.019 if joint_index == 0 else 0.014)
                end = (start[0], start[1], start[2] - length)
                previous = add_bone(
                    bones, f"{side}{finger}{joint_name}",
                    start, end, previous,
                )
                start = end

        hip_x = center_x + sign * width * 0.09
        knee_x = center_x + sign * width * 0.10
        ankle_x = center_x + sign * width * 0.10
        upper_leg = add_bone(
            bones, f"{side}UpperLeg",
            (hip_x, center_y, at(0.51)),
            (knee_x, center_y, at(0.28)),
            hips,
        )
        lower_leg = add_bone(
            bones, f"{side}LowerLeg",
            upper_leg.tail,
            (ankle_x, center_y, at(0.06)),
            upper_leg,
        )
        add_bone(
            bones, f"{side}Foot",
            lower_leg.tail,
            (ankle_x, center_y + height * 0.07, at(0.035)),
            lower_leg,
        )

    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def has_complete_weights(mesh, armature):
    deform_names = {bone.name for bone in armature.data.bones if bone.use_deform}
    indices = {group.index for group in mesh.vertex_groups if group.name in deform_names}
    return bool(mesh.data.vertices) and all(
        any(group.group in indices and math.isfinite(group.weight) and group.weight > 0
            for group in vertex.groups)
        for vertex in mesh.data.vertices
    )


def segment_distance_squared(point, head, tail):
    direction = tail - head
    fraction = ((point - head).dot(direction) / direction.length_squared
                if direction.length_squared > 1e-20 else 0.0)
    closest = head + direction * max(0.0, min(1.0, fraction))
    return (point - closest).length_squared


def proximity_weights(mesh, armature):
    """Coarse fallback: blend the nearest body bone with an adjacent bone.

    Finger landmarks cannot be recovered from this mesh, so fallback weights
    use the hand bones rather than pretending to recover finger articulation.
    """
    finger_parts = ("Thumb", "Index", "Middle", "Ring", "Little")
    bones = [
        bone for bone in armature.data.bones
        if bone.use_deform and not any(part in bone.name for part in finger_parts)
    ]
    if not bones:
        fail("No deform bones are available for approximate weighting.")
    mesh.vertex_groups.clear()
    groups = {bone.name: mesh.vertex_groups.new(name=bone.name) for bone in bones}
    inverse = armature.matrix_world.inverted()
    for vertex in mesh.data.vertices:
        point = inverse @ mesh.matrix_world @ vertex.co
        distances = {
            bone.name: segment_distance_squared(point, bone.head_local, bone.tail_local)
            for bone in bones
        }
        nearest = min(bones, key=lambda bone: distances[bone.name])
        # Adjacent bones only: avoid blending across the two legs or arms.
        candidates = [
            bone for bone in bones
            if bone == nearest or bone.parent == nearest or nearest.parent == bone
        ]
        selected = sorted(candidates, key=lambda bone: distances[bone.name])[:2]
        weights = [1.0 / max(distances[bone.name], 1e-12) for bone in selected]
        total = sum(weights)
        for bone, weight in zip(selected, weights):
            groups[bone.name].add([vertex.index], weight / total, "REPLACE")
    world = mesh.matrix_world.copy()
    mesh.parent = armature
    mesh.matrix_world = world
    modifiers = [modifier for modifier in mesh.modifiers if modifier.type == "ARMATURE"]
    modifier = modifiers[0] if modifiers else mesh.modifiers.new("AvatarArmature", "ARMATURE")
    modifier.object = armature
    modifier.use_vertex_groups = True


def bind_meshes(meshes, armature):
    used_fallback = False
    for mesh in meshes:
        bpy.ops.object.select_all(action="DESELECT")
        mesh.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        auto_failed = False
        try:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        except RuntimeError as exc:
            print(f"RIGGING_WARNING: Automatic weights failed for {mesh.name}: {exc}")
            auto_failed = True
        if auto_failed or not has_complete_weights(mesh, armature):
            print(f"RIGGING_WARNING: Using approximate proximity weights for {mesh.name}; "
                  "joint bending may need manual adjustment and fingers are not articulated.")
            proximity_weights(mesh, armature)
            used_fallback = True
        if not has_complete_weights(mesh, armature):
            fail(f"Mesh {mesh.name} still has vertices without usable skin weights.")
    return used_fallback


def _read_accessor_rows(document, binary, accessor_index):
    accessor = document["accessors"][accessor_index]
    component_formats = {
        5121: ("B", 1, 255),
        5123: ("H", 2, 65535),
        5126: ("f", 4, None),
    }
    widths = {"VEC4": 4}
    component_type = accessor.get("componentType")
    if component_type not in component_formats or accessor.get("type") not in widths:
        fail("Exported skin attributes use an unsupported component type.")
    fmt, component_size, normalization_max = component_formats[component_type]
    width = widths[accessor["type"]]
    element_size = component_size * width
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        fail("Exported skin attributes refer to a non-GLB buffer.")
    stride = view.get("byteStride", element_size)
    if stride < element_size:
        fail("Exported skin attribute stride is invalid.")
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    for row_index in range(accessor["count"]):
        row_offset = offset + row_index * stride
        if row_offset < 0 or row_offset + element_size > len(binary):
            fail("Exported skin attribute data is truncated.")
        row = struct.unpack_from("<" + fmt * width, binary, row_offset)
        if accessor.get("normalized") and normalization_max:
            row = tuple(value / normalization_max for value in row)
        yield row


def verify_export(output):
    """Check for a real skin and usable influence on every required body joint."""
    data = output.read_bytes()
    if len(data) < 20:
        fail("Exported GLB is truncated.")
    magic, version, length, chunk_length, chunk_type = struct.unpack_from("<4sIIII", data)
    if (magic != b"glTF" or version != 2 or length != len(data)
            or chunk_type != 0x4E4F534A or 20 + chunk_length > len(data)):
        fail("Exported GLB has an invalid header.")
    document = None
    binary = b""
    offset = 12
    while offset + 8 <= len(data):
        current_length, current_type = struct.unpack_from("<II", data, offset)
        start, end = offset + 8, offset + 8 + current_length
        if end > len(data):
            fail("Exported GLB has a truncated chunk.")
        if current_type == 0x4E4F534A:
            document = json.loads(data[start:end].decode("utf-8").rstrip(" \t\r\n\0"))
        elif current_type == 0x004E4942:
            binary = data[start:end]
        offset = end
    if document is None:
        fail("Exported GLB has no JSON document.")
    skins = document.get("skins", [])
    meshes = document.get("meshes", [])
    mesh_nodes = [node for node in document.get("nodes", []) if "mesh" in node]
    if not mesh_nodes:
        fail("Exported GLB has no mesh nodes.")
    weighted_joint_counts = {}
    for node in mesh_nodes:
        skin = node.get("skin")
        if not isinstance(skin, int) or not 0 <= skin < len(skins) or not skins[skin].get("joints"):
            fail("Exported mesh has no usable skin binding.")
        mesh_index = node["mesh"]
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            fail("Exported mesh node refers to a missing mesh.")
        primitives = meshes[mesh_index].get("primitives", [])
        if not primitives:
            fail("Exported mesh has no primitives.")
        joint_nodes = skins[skin]["joints"]
        for primitive in primitives:
            attributes = primitive.get("attributes", {})
            if not {"JOINTS_0", "WEIGHTS_0"}.issubset(attributes):
                fail("Exported mesh has no joint/weight attributes.")
            if (
                document["accessors"][attributes["JOINTS_0"]]["count"]
                != document["accessors"][attributes["WEIGHTS_0"]]["count"]
            ):
                fail("Exported mesh joint and weight counts do not match.")
            joint_rows = _read_accessor_rows(document, binary, attributes["JOINTS_0"])
            weight_rows = _read_accessor_rows(document, binary, attributes["WEIGHTS_0"])
            for joint_row, weight_row in zip(joint_rows, weight_rows):
                if any(not math.isfinite(weight) or weight < 0 for weight in weight_row):
                    fail("Exported mesh contains invalid vertex weights.")
                if sum(weight_row) <= 1e-8:
                    fail("Exported mesh contains vertices with no usable skin weights.")
                for joint_index, weight in zip(joint_row, weight_row):
                    if weight <= 0:
                        continue
                    if not isinstance(joint_index, int) or not 0 <= joint_index < len(joint_nodes):
                        fail("Exported mesh contains an invalid skin joint index.")
                    node_index = joint_nodes[joint_index]
                    if not isinstance(node_index, int) or not 0 <= node_index < len(document["nodes"]):
                        fail("Exported skin refers to a missing joint node.")
                    joint_name = document["nodes"][node_index].get("name")
                    if joint_name:
                        weighted_joint_counts[joint_name] = (
                            weighted_joint_counts.get(joint_name, 0) + 1
                        )
    missing_weights = sorted(
        REQUIRED_WEIGHTED_JOINTS - weighted_joint_counts.keys()
    )
    if missing_weights:
        fail(
            "Exported skin has no vertex weights for required body joints: "
            + ", ".join(missing_weights)
        )


def main():
    source, output = parse_arguments()
    if not source.is_file():
        fail(f"Input GLB does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        fail("The input GLB contains no mesh geometry.")
    bounds = mesh_bounds(meshes)
    armature = create_humanoid_armature(bounds)
    used_fallback = bind_meshes(meshes, armature)
    armature["weighting_method"] = "approximate_proximity" if used_fallback else "automatic_heat"
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_skins=True,
        export_apply=False,
        export_extras=True,
    )
    if not output.is_file():
        fail("Blender did not create the rigged GLB.")
    verify_export(output)
    print(f"RIGGING_SUCCESS: {output}")


if __name__ == "__main__":
    main()
