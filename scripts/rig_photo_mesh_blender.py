"""Bind a full-body GLB to a simple humanoid armature using Blender's local API."""

import sys
import json
import math
import struct
from pathlib import Path

import bpy
from mathutils import Vector


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


def verify_export(output):
    """Blender may export successfully even when the mesh has no skin."""
    data = output.read_bytes()
    if len(data) < 20:
        fail("Exported GLB is truncated.")
    magic, version, length, chunk_length, chunk_type = struct.unpack_from("<4sIIII", data)
    if (magic != b"glTF" or version != 2 or length != len(data)
            or chunk_type != 0x4E4F534A or 20 + chunk_length > len(data)):
        fail("Exported GLB has an invalid header.")
    document = json.loads(data[20:20 + chunk_length])
    skins = document.get("skins", [])
    meshes = document.get("meshes", [])
    mesh_nodes = [node for node in document.get("nodes", []) if "mesh" in node]
    if not mesh_nodes:
        fail("Exported GLB has no mesh nodes.")
    for node in mesh_nodes:
        skin = node.get("skin")
        if not isinstance(skin, int) or not 0 <= skin < len(skins) or not skins[skin].get("joints"):
            fail("Exported mesh has no usable skin binding.")
        primitives = meshes[node["mesh"]].get("primitives", [])
        if not primitives or not all(
            {"JOINTS_0", "WEIGHTS_0"}.issubset(primitive.get("attributes", {}))
            for primitive in primitives
        ):
            fail("Exported mesh has no joint/weight attributes.")


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
