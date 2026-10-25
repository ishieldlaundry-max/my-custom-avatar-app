from pathlib import Path
import shutil
import subprocess

import pytest


LIVE_AVATAR = Path("assets/live_avatar.html")


def _extract_function(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Could not find the end of {name}()")


def _node():
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node.js is required to execute the live tracker helpers")
    return executable


def test_tracking_status_only_reports_detected_avatar_parts():
    source = LIVE_AVATAR.read_text(encoding="utf-8")
    helper = _extract_function(source, "formatTrackingStatus")
    script = (
        helper
        + """
const cases = [
  formatTrackingStatus(true, true, 0),
  formatTrackingStatus(true, false, 0),
  formatTrackingStatus(false, true, 0),
  formatTrackingStatus(false, false, 1),
  formatTrackingStatus(false, false, 2),
  formatTrackingStatus(false, false, 0)
];
process.stdout.write(JSON.stringify(cases));
"""
    )
    result = subprocess.run(
        [_node(), "--input-type=commonjs", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == (
        '["FACE + BODY","FACE","BODY","1 HAND","2 HANDS","NO TRACK"]'
    )


def test_hip_roll_is_bounded_and_ignores_unreliable_landmarks():
    source = LIVE_AVATAR.read_text(encoding="utf-8")
    helper = _extract_function(source, "getHipRollTarget")
    script = (
        helper
        + """
const pose = (left, right) => {
  const landmarks = [];
  landmarks[23] = left;
  landmarks[24] = right;
  return getHipRollTarget(landmarks);
};
const result = [
  pose({x: 0.4, y: 0.5, visibility: 1}, {x: 0.6, y: 0.5, visibility: 1}),
  pose({x: 0.4, y: 0.5, visibility: 1}, {x: 0.6, y: 0.7, visibility: 1}),
  pose({x: 0.4, y: 0.7, visibility: 1}, {x: 0.6, y: 0.5, visibility: 1}),
  pose({x: 0.4, y: 0.5, visibility: 0.2}, {x: 0.6, y: 0.5, visibility: 1}),
  pose({x: NaN, y: 0.5, visibility: 1}, {x: 0.6, y: 0.5, visibility: 1})
];
process.stdout.write(JSON.stringify(result));
"""
    )
    result = subprocess.run(
        [_node(), "--input-type=commonjs", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    values = __import__("json").loads(result.stdout)
    assert values[:3] == pytest.approx([0, 0.35, -0.35])
    assert values[3:] == [None, None]


def test_pose_application_drives_pelvis_and_reports_actual_tracking_state():
    source = LIVE_AVATAR.read_text(encoding="utf-8")
    pose_code = source.split("function applyPose(landmarks) {", 1)[1].split(
        "function applyHands(", 1
    )[0]
    frame_code = source.split("function trackFrame() {", 1)[1].split(
        '$("start").addEventListener', 1
    )[0]

    assert "const hip = getBone(boneNames.hips);" in pose_code
    assert "smooth(hip, \"z\", hipRoll);" in pose_code
    assert "return trackedSegments > 0;" in pose_code
    assert "const bodyTracked = applyPose(poseLandmarks);" in frame_code
    assert "const faceTracked = applyFace(face);" in frame_code
    assert "trackingEl.textContent = formatTrackingStatus(" in frame_code
    assert 'trackingEl.textContent = "FACE + BODY";' not in source


def test_live_avatar_module_has_valid_javascript_syntax(tmp_path):
    source = LIVE_AVATAR.read_text(encoding="utf-8")
    module = source.split('<script type="module">', 1)[1].split("</script>", 1)[0]
    path = tmp_path / "live_avatar.mjs"
    path.write_text(module, encoding="utf-8")
    subprocess.run([_node(), "--check", str(path)], check=True, capture_output=True, text=True)
