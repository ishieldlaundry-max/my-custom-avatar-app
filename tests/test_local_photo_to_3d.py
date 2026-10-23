from pathlib import Path
from types import SimpleNamespace
import json
import struct

import pytest

from src.pipelines.local_photo_to_3d import (
    LocalPhotoAvatarResult,
    LocalPhotoMeshResult,
    LocalPhotoTo3DError,
    find_blender_executable,
    generate_local_photo_mesh,
    generate_rigged_local_photo_avatar,
    rig_local_photo_mesh,
    sf3d_python_path,
    validate_texture_resolution,
)


@pytest.mark.parametrize("resolution", [512, 768, 1024])
def test_supported_texture_resolutions(resolution):
    assert validate_texture_resolution(resolution) == resolution


@pytest.mark.parametrize("resolution", [256, 640, 2048, "large", None])
def test_rejects_unsupported_texture_resolutions(resolution):
    with pytest.raises(LocalPhotoTo3DError, match="512, 768, or 1024"):
        validate_texture_resolution(resolution)


def test_rejects_missing_source_image(tmp_path):
    with pytest.raises(LocalPhotoTo3DError, match="Choose an image file"):
        generate_local_photo_mesh(tmp_path / "missing.jpg", sf3d_root=tmp_path)


def test_explains_missing_local_install(tmp_path):
    image = tmp_path / "portrait.jpg"
    image.write_bytes(b"not decoded by the launcher")
    with pytest.raises(LocalPhotoTo3DError, match="Stable Fast 3D is not installed"):
        generate_local_photo_mesh(
            image,
            sf3d_root=tmp_path / "missing-sf3d",
            python_path=tmp_path / "missing-python.exe",
        )


def test_windows_uses_separate_sf3d_interpreter(monkeypatch, tmp_path):
    import src.pipelines.local_photo_to_3d as pipeline

    monkeypatch.setattr(pipeline.os, "name", "nt")
    assert sf3d_python_path(tmp_path) == tmp_path / ".venv" / "Scripts" / "python.exe"


def test_explains_when_blender_is_not_installed(monkeypatch):
    import src.pipelines.local_photo_to_3d as pipeline

    monkeypatch.delenv("BLENDER_EXECUTABLE", raising=False)
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: None)
    with pytest.raises(LocalPhotoTo3DError, match="Install the free Blender"):
        find_blender_executable()


def test_rigged_glb_uses_local_blender_and_keeps_source(monkeypatch, tmp_path):
    import src.pipelines.local_photo_to_3d as pipeline

    source = tmp_path / "mesh.glb"
    make_glb(source, skinned=False)
    blender_script = tmp_path / "rig.py"
    blender_script.write_text("# Blender fixture")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"executable fixture")
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        make_glb(Path(command[-1]), skinned=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline, "find_blender_executable", lambda _path=None: blender)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = rig_local_photo_mesh(source, script_path=blender_script)

    assert Path(result).name == "mesh_rigged.glb"
    assert Path(result).read_bytes().startswith(b"glTF")
    assert not pipeline.read_glb_document(source).get("skins")
    command, kwargs = commands[0]
    assert command[:4] == [str(blender), "--background", "--python", str(blender_script)]
    assert kwargs["timeout"] == 900


def test_rig_validation_reports_failed_checks_and_keeps_candidate(monkeypatch, tmp_path):
    import src.pipelines.local_photo_to_3d as pipeline

    source = tmp_path / "mesh.glb"
    make_glb(source, skinned=False)
    blender_script = tmp_path / "rig.py"
    blender_script.write_text("# Blender fixture")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"executable fixture")

    def fake_run(command, **kwargs):
        make_glb(Path(command[-1]), skinned=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline, "find_blender_executable", lambda _path=None: blender)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    candidate = tmp_path / "mesh_rigged.glb"
    with pytest.raises(LocalPhotoTo3DError) as exc_info:
        rig_local_photo_mesh(source, script_path=blender_script)

    message = str(exc_info.value)
    assert "rig validation failed" in message
    assert "the exported GLB has no skin" in message
    assert "JOINTS_0 and WEIGHTS_0" in message
    assert "required humanoid joints are missing" in message
    assert str(candidate) in message
    assert candidate.is_file()
    assert source.is_file()


def test_photo_avatar_generation_automatically_rigs_the_mesh(monkeypatch, tmp_path):
    import src.pipelines.local_photo_to_3d as pipeline

    source_image = tmp_path / "portrait.jpg"
    source_mesh = tmp_path / "mesh.glb"
    rigged_model = tmp_path / "mesh_rigged.glb"
    source_image.write_bytes(b"image")
    source_mesh.write_bytes(b"mesh")
    rigged_model.write_bytes(b"rigged")
    calls = []

    def fake_generate(image, resolution, **kwargs):
        calls.append(("generate", image, resolution, kwargs))
        return LocalPhotoMeshResult(str(source_mesh), 42.5, 6144.0)

    def fake_rig(model, **kwargs):
        calls.append(("rig", model, kwargs))
        return str(rigged_model)

    monkeypatch.setattr(pipeline, "generate_local_photo_mesh", fake_generate)
    monkeypatch.setattr(pipeline, "rig_local_photo_mesh", fake_rig)

    result = generate_rigged_local_photo_avatar(
        source_image,
        768,
        sf3d_root=tmp_path / "sf3d",
        python_path=tmp_path / "python.exe",
        blender_path=tmp_path / "blender.exe",
    )

    assert result == LocalPhotoAvatarResult(
        model_path=str(rigged_model),
        source_mesh_path=str(source_mesh),
        reconstruction_seconds=42.5,
        peak_allocated_vram_mb=6144.0,
    )
    assert calls[0][0:3] == ("generate", source_image, 768)
    assert calls[1][0:2] == ("rig", str(source_mesh))
    assert calls[1][2]["blender_path"] == tmp_path / "blender.exe"


def test_automatic_photo_rigging_error_preserves_the_generated_mesh(monkeypatch, tmp_path):
    import src.pipelines.local_photo_to_3d as pipeline

    source_mesh = tmp_path / "mesh.glb"
    candidate = tmp_path / "mesh_rigged.glb"
    source_mesh.write_bytes(b"mesh")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(
        pipeline,
        "generate_local_photo_mesh",
        lambda *_args, **_kwargs: LocalPhotoMeshResult(
            str(source_mesh), 12.0, None
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "rig_local_photo_mesh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LocalPhotoTo3DError(
                f"rig validation failed; candidate file was kept at {candidate}"
            )
        ),
    )

    with pytest.raises(LocalPhotoTo3DError) as exc_info:
        generate_rigged_local_photo_avatar(tmp_path / "portrait.jpg")

    message = str(exc_info.value)
    assert "automatic avatar preparation could not finish" in message
    assert str(source_mesh) in message
    assert str(candidate) in message
    assert source_mesh.is_file()
    assert candidate.is_file()


def make_glb(path, *, skinned):
    joint_names = [
        "Hips", "Spine", "Chest", "Neck", "Head",
        "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm",
        "LeftUpperLeg", "LeftLowerLeg", "RightUpperLeg", "RightLowerLeg",
    ]
    document = {
        "asset": {"version": "2.0"},
        "images": [{"uri": "texture.png"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    if skinned:
        document["nodes"] = [{"name": name} for name in joint_names]
        document["skins"] = [{"joints": list(range(len(joint_names)))}]
        document["meshes"][0]["primitives"][0]["attributes"].update(
            {"JOINTS_0": 1, "WEIGHTS_0": 2}
        )
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    chunk = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    total_length = 12 + len(chunk)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, total_length) + chunk)


def test_rejects_non_glb_rigging_input(tmp_path):
    model = tmp_path / "mesh.obj"
    model.write_text("mesh")
    with pytest.raises(LocalPhotoTo3DError, match="GLB files only"):
        rig_local_photo_mesh(model)


def test_generation_returns_runtime_and_peak_memory(monkeypatch, tmp_path):
    import src.pipelines.local_photo_to_3d as pipeline

    image = tmp_path / "portrait.jpg"
    image.write_bytes(b"image placeholder")
    root = tmp_path / "sf3d"
    root.mkdir()
    (root / "run.py").write_text("# SF3D runner placeholder")
    interpreter = tmp_path / "python.exe"
    interpreter.write_bytes(b"python placeholder")

    def fake_run(command, **kwargs):
        if "-c" in command:
            return SimpleNamespace(returncode=0, stdout="CUDA\n", stderr="")
        output_dir = Path(command[command.index("--output-dir") + 1])
        mesh_path = output_dir / "0" / "mesh.glb"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_bytes(b"glb placeholder")
        return SimpleNamespace(
            returncode=0,
            stdout="Peak Memory: 6144.5 MB\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = generate_local_photo_mesh(
        image,
        512,
        sf3d_root=root,
        python_path=interpreter,
    )
    assert Path(result.model_path).is_file()
    assert result.elapsed_seconds >= 0
    assert result.peak_allocated_vram_mb == 6144.5
