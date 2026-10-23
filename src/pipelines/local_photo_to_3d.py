"""Optional local Stable Fast 3D photo reconstruction for Windows/NVIDIA."""

from __future__ import annotations

import os
import json
import re
import shutil
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class LocalPhotoTo3DError(RuntimeError):
    """A local photo reconstruction could not be completed."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SF3D_ROOT = PROJECT_ROOT / "third_party" / "stable-fast-3d"
SF3D_RUNNER = SF3D_ROOT / "run.py"
BLENDER_RIG_SCRIPT = PROJECT_ROOT / "scripts" / "rig_photo_mesh_blender.py"
GLB_JSON_CHUNK = 0x4E4F534A


def read_glb_document(path: Path) -> dict:
    """Read the JSON chunk from a binary GLB for lightweight output checks."""
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("GLB is too small to contain a JSON document.")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError("GLB header is invalid.")
    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        start, end = offset + 8, offset + 8 + chunk_length
        if end > len(data):
            raise ValueError("GLB chunk length exceeds the file size.")
        if chunk_type == GLB_JSON_CHUNK:
            return json.loads(data[start:end].decode("utf-8").rstrip(" \t\r\n\0"))
        offset = end
    raise ValueError("GLB has no JSON document chunk.")


@dataclass(frozen=True)
class LocalPhotoMeshResult:
    model_path: str
    elapsed_seconds: float
    peak_allocated_vram_mb: float | None


@dataclass(frozen=True)
class LocalPhotoAvatarResult:
    model_path: str
    source_mesh_path: str
    reconstruction_seconds: float
    peak_allocated_vram_mb: float | None


def find_blender_executable(blender_path: str | Path | None = None) -> Path:
    """Find a local Blender install without downloading or installing anything."""
    configured_path = blender_path or os.environ.get("BLENDER_EXECUTABLE")
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        located = shutil.which(str(configured_path))
        if located:
            return Path(located).resolve()
        raise LocalPhotoTo3DError(
            "The configured Blender executable was not found. Set "
            "BLENDER_EXECUTABLE to Blender's executable path."
        )

    located = shutil.which("blender")
    if located:
        return Path(located).resolve()

    if os.name == "nt":
        program_files = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramW6432", r"C:\Program Files")),
        ]
        candidates = [
            executable
            for root in program_files
            for executable in (root / "Blender Foundation").glob("Blender */blender.exe")
        ]
        if candidates:
            return sorted(candidates, key=lambda path: path.parent.name)[-1].resolve()

    raise LocalPhotoTo3DError(
        "Blender was not found. Install the free Blender desktop app, then retry. "
        "The original textured GLB is unchanged."
    )


def rig_local_photo_mesh(
    model_path: str | Path,
    *,
    blender_path: str | Path | None = None,
    script_path: Path = BLENDER_RIG_SCRIPT,
    timeout_seconds: int = 900,
) -> str:
    """Bind a generated GLB to a coarse humanoid skeleton using local Blender."""
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise LocalPhotoTo3DError("Choose a generated GLB file that still exists.")
    if source.suffix.lower() != ".glb":
        raise LocalPhotoTo3DError("The local rigging step accepts GLB files only.")
    if not script_path.is_file():
        raise LocalPhotoTo3DError("The local Blender rigging script is missing.")

    blender = find_blender_executable(blender_path)
    output = source.with_name(f"{source.stem}_rigged.glb")
    command = [
        str(blender),
        "--background",
        "--python",
        str(script_path),
        "--",
        str(source),
        str(output),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalPhotoTo3DError(
            "Blender rigging exceeded 15 minutes and was stopped. "
            "The original textured GLB is unchanged."
        ) from exc

    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout or "No diagnostic output was returned.")
        detail = "\n".join(detail.strip().splitlines()[-12:])
        raise LocalPhotoTo3DError(
            "Blender could not bind the mesh to a humanoid skeleton. Try an upright, "
            "full-body photo with clear space around the arms and legs. The original "
            "textured GLB is unchanged.\n" + detail
        )
    try:
        source_document = read_glb_document(source)
        output_document = read_glb_document(output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        output.unlink(missing_ok=True)
        raise LocalPhotoTo3DError(f"Blender produced an invalid GLB: {exc}") from exc
    has_skin = bool(output_document.get("skins"))
    has_weighted_mesh = any(
        "JOINTS_0" in primitive.get("attributes", {})
        and "WEIGHTS_0" in primitive.get("attributes", {})
        for mesh in output_document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    )
    source_image_count = len(source_document.get("images", []))
    output_image_count = len(output_document.get("images", []))
    source_texture_count = len(source_document.get("textures", []))
    output_texture_count = len(output_document.get("textures", []))
    required_joints = {
        "Hips", "Spine", "Chest", "Neck", "Head",
        "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm",
        "LeftUpperLeg", "LeftLowerLeg", "RightUpperLeg", "RightLowerLeg",
    }
    output_joint_names = {
        node.get("name")
        for node in output_document.get("nodes", [])
        if isinstance(node.get("name"), str)
    }
    if (
        not has_skin
        or not has_weighted_mesh
        or not required_joints.issubset(output_joint_names)
        or output_image_count < source_image_count
        or output_texture_count < source_texture_count
    ):
        validation_issues = []
        if not has_skin:
            validation_issues.append("the exported GLB has no skin")
        if not has_weighted_mesh:
            validation_issues.append(
                "no exported mesh has both JOINTS_0 and WEIGHTS_0 attributes"
            )
        missing_joints = sorted(required_joints - output_joint_names)
        if missing_joints:
            validation_issues.append(
                "required humanoid joints are missing: " + ", ".join(missing_joints)
            )
        if output_image_count < source_image_count:
            validation_issues.append(
                f"embedded image count decreased from {source_image_count} "
                f"to {output_image_count}"
            )
        if output_texture_count < source_texture_count:
            validation_issues.append(
                f"embedded texture count decreased from {source_texture_count} "
                f"to {output_texture_count}"
            )
        raise LocalPhotoTo3DError(
            "Blender produced a GLB, but rig validation failed: "
            + "; ".join(validation_issues)
            + f". The candidate file was kept at {output}. "
            "The original textured GLB is unchanged."
        )
    return str(output)


def sf3d_python_path(root: Path = SF3D_ROOT) -> Path:
    """Return the isolated SF3D interpreter path for the current platform."""
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def validate_texture_resolution(texture_resolution: int) -> int:
    """Keep texture baking within tested, predictable output sizes."""
    try:
        resolution = int(texture_resolution)
    except (TypeError, ValueError) as exc:
        raise LocalPhotoTo3DError("Choose a texture resolution of 512, 768, or 1024.") from exc
    if resolution not in {512, 768, 1024}:
        raise LocalPhotoTo3DError("Choose a texture resolution of 512, 768, or 1024.")
    return resolution


def generate_local_photo_mesh(
    source_image: str | Path,
    texture_resolution: int = 512,
    *,
    sf3d_root: Path = SF3D_ROOT,
    python_path: Path | None = None,
) -> LocalPhotoMeshResult:
    """Generate a textured, static GLB using an isolated local SF3D install."""
    resolution = validate_texture_resolution(texture_resolution)
    image_path = Path(source_image).expanduser().resolve()
    if not image_path.is_file():
        raise LocalPhotoTo3DError("Choose an image file that still exists on this computer.")

    runner = sf3d_root / "run.py"
    interpreter = python_path or sf3d_python_path(sf3d_root)
    if not runner.is_file() or not interpreter.is_file():
        raise LocalPhotoTo3DError(
            "Stable Fast 3D is not installed. Follow the local Windows setup guide "
            "for this app, request access to the Stable Fast 3D model, and sign in "
            "to Hugging Face from the isolated SF3D environment."
        )

    cuda_check = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import torch; print('CUDA' if torch.cuda.is_available() else 'NO_CUDA')",
        ],
        cwd=sf3d_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if cuda_check.returncode != 0:
        diagnostic = (cuda_check.stderr or cuda_check.stdout).strip().splitlines()[-1:]
        raise LocalPhotoTo3DError(
            "The isolated SF3D Python environment cannot load PyTorch. "
            + (" ".join(diagnostic) if diagnostic else "Check the setup guide.")
        )
    if "CUDA" not in cuda_check.stdout or "NO_CUDA" in cuda_check.stdout:
        raise LocalPhotoTo3DError(
            "PyTorch in the SF3D environment cannot see a CUDA GPU. "
            "The app stopped rather than running this reconstruction on the CPU."
        )

    output_dir = PROJECT_ROOT / "results" / "local_photo_to_3d" / uuid.uuid4().hex
    command = [
        str(interpreter),
        str(runner),
        str(image_path),
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
        "--texture-resolution",
        str(resolution),
        "--remesh_option",
        "none",
    ]
    started_at = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=sf3d_root,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalPhotoTo3DError(
            "Local reconstruction exceeded 30 minutes and was stopped."
        ) from exc
    elapsed_seconds = time.perf_counter() - started_at

    model_path = output_dir / "0" / "mesh.glb"
    if result.returncode != 0 or not model_path.is_file():
        detail = (result.stderr or result.stdout or "No diagnostic output was returned.")
        detail = "\n".join(detail.strip().splitlines()[-12:])
        raise LocalPhotoTo3DError(
            "Stable Fast 3D did not produce a textured model. "
            "Check that model access is approved and GPU memory is available.\n"
            + detail
        )

    output_log = f"{result.stdout}\n{result.stderr}"
    peak_match = re.search(r"Peak Memory:\s*([\d.]+)\s*MB", output_log)
    peak_memory = float(peak_match.group(1)) if peak_match else None
    return LocalPhotoMeshResult(
        model_path=str(model_path),
        elapsed_seconds=elapsed_seconds,
        peak_allocated_vram_mb=peak_memory,
    )


def generate_rigged_local_photo_avatar(
    source_image: str | Path,
    texture_resolution: int = 512,
    *,
    sf3d_root: Path = SF3D_ROOT,
    python_path: Path | None = None,
    blender_path: str | Path | None = None,
    script_path: Path = BLENDER_RIG_SCRIPT,
) -> LocalPhotoAvatarResult:
    """Generate and rig a photo avatar in one user-facing operation."""
    mesh = generate_local_photo_mesh(
        source_image,
        texture_resolution,
        sf3d_root=sf3d_root,
        python_path=python_path,
    )
    try:
        rigged_path = rig_local_photo_mesh(
            mesh.model_path,
            blender_path=blender_path,
            script_path=script_path,
        )
    except LocalPhotoTo3DError as exc:
        raise LocalPhotoTo3DError(
            "Photo reconstruction succeeded, but automatic avatar preparation "
            f"could not finish. The textured mesh is preserved at {mesh.model_path}. "
            f"{exc}"
        ) from exc
    return LocalPhotoAvatarResult(
        model_path=rigged_path,
        source_mesh_path=mesh.model_path,
        reconstruction_seconds=mesh.elapsed_seconds,
        peak_allocated_vram_mb=mesh.peak_allocated_vram_mb,
    )
