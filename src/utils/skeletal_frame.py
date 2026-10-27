"""Skeletal frame contract and DWPose landmark normalization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Optional

import numpy as np


BODY_LANDMARK_NAMES = (
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)

BODY_BONES = {
    "Head": (1, 0),
    "RightUpperArm": (2, 3),
    "RightLowerArm": (3, 4),
    "LeftUpperArm": (5, 6),
    "LeftLowerArm": (6, 7),
    "RightUpperLeg": (8, 9),
    "RightLowerLeg": (9, 10),
    "LeftUpperLeg": (11, 12),
    "LeftLowerLeg": (12, 13),
}

HAND_FINGERS = (
    ("Thumb", (1, 2, 3, 4), ("Proximal", "Distal")),
    ("Index", (0, 5, 6, 7, 8), ("Proximal", "Intermediate", "Distal")),
    ("Middle", (0, 9, 10, 11, 12), ("Proximal", "Intermediate", "Distal")),
    ("Ring", (0, 13, 14, 15, 16), ("Proximal", "Intermediate", "Distal")),
    ("Little", (0, 17, 18, 19, 20), ("Proximal", "Intermediate", "Distal")),
)

BODY_CONNECTIONS = (
    ("nose", "neck"),
    ("neck", "right_shoulder"),
    ("neck", "left_shoulder"),
    ("right_shoulder", "left_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_hip", "left_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
)


@dataclass(frozen=True)
class SkeletalFrame:
    """Serializable pose sample; coordinates are body-height-relative 2.5D."""

    timestamp_ms: int
    tracked: bool
    calibrated: bool
    root_translation: tuple[float, float, float]
    landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    image_landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    bones: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    blendshapes: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    foot_contacts: dict[str, bool] = field(default_factory=dict)
    depth_mode: str = "uncalibrated"

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation without NumPy values."""
        result = asdict(self)
        result["root_translation"] = list(self.root_translation)
        result["landmarks"] = {
            key: list(value) for key, value in self.landmarks.items()
        }
        result["image_landmarks"] = {
            key: list(value) for key, value in self.image_landmarks.items()
        }
        result["bones"] = {
            key: list(value) for key, value in self.bones.items()
        }
        return result


@dataclass
class _Person:
    body: np.ndarray
    face: np.ndarray
    hands: dict[str, np.ndarray]
    body_confidence: np.ndarray
    face_confidence: np.ndarray
    hand_confidence: dict[str, np.ndarray]


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _distance(a, b) -> float:
    return float(np.linalg.norm(
        np.asarray(a, dtype=np.float32)[:2] -
        np.asarray(b, dtype=np.float32)[:2]
    ))


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _angle_2d(a, b) -> float:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    return math.atan2(-dy, dx)


def _joint_angle(a, b, c) -> float:
    first = np.asarray(a[:2], dtype=np.float64) - np.asarray(b[:2], dtype=np.float64)
    second = np.asarray(c[:2], dtype=np.float64) - np.asarray(b[:2], dtype=np.float64)
    lengths = float(np.linalg.norm(first) * np.linalg.norm(second))
    if lengths < 1e-8:
        return 0.0
    cosine = _clamp(float(np.dot(first, second) / lengths), -1.0, 1.0)
    return math.pi - math.acos(cosine)


def _pose_people(pose: dict) -> list[_Person]:
    """Normalize DWPose Wholebody output into per-person arrays."""
    bodies = pose.get("bodies") or {}
    subset = np.asarray(bodies.get("subset", []), dtype=np.float32)
    if subset.size == 0:
        return []
    if subset.ndim == 1:
        subset = subset[None, :]
    count = len(subset)
    candidate = np.asarray(bodies.get("candidate", []), dtype=np.float32)
    if candidate.size == 0:
        return []
    candidate = candidate.reshape((count, 18, -1))
    if candidate.shape[2] < 3:
        scores = np.asarray(bodies.get("score", []), dtype=np.float32)
        candidate = np.concatenate(
            [candidate[:, :, :2], scores.reshape((count, 18, 1))], axis=2
        )
    body_confidence = np.asarray(
        bodies.get("score", candidate[:, :, 2]), dtype=np.float32
    )
    if body_confidence.size == 0:
        body_confidence = candidate[:, :, 2]
    body_confidence = body_confidence.reshape((count, 18))

    faces = np.asarray(pose.get("faces", []), dtype=np.float32)
    face_scores = np.asarray(pose.get("faces_score", []), dtype=np.float32)
    hands = np.asarray(pose.get("hands", []), dtype=np.float32)
    hand_scores = np.asarray(pose.get("hands_score", []), dtype=np.float32)
    if faces.size:
        faces = faces.reshape((count, 68, -1))
    if face_scores.size:
        face_scores = face_scores.reshape((count, 68))
    if hands.size:
        hands = hands.reshape((2 * count, 21, -1))
    if hand_scores.size:
        hand_scores = hand_scores.reshape((2 * count, 21))

    people = []
    for person_index in range(count):
        body = candidate[person_index, :, :3].copy()
        body[:, 2] = np.clip(body_confidence[person_index], 0.0, 1.0)
        face = (
            faces[person_index, :, :3].copy()
            if faces.size
            else np.zeros((68, 3), dtype=np.float32)
        )
        if face_scores.size:
            face[:, 2] = np.clip(face_scores[person_index], 0.0, 1.0)
        face_confidence = face[:, 2].copy()

        hand_candidates = []
        for hand_index in (person_index, count + person_index):
            if hands.size and hand_index < len(hands):
                hand = hands[hand_index, :, :3].copy()
                scores = (
                    hand_scores[hand_index].copy()
                    if hand_scores.size
                    else hand[:, 2].copy()
                )
                hand[:, 2] = np.clip(scores, 0.0, 1.0)
                if np.max(hand[:, 2]) > 0.05:
                    hand_candidates.append((hand, hand[:, 2].copy()))

        assigned_hands: dict[str, np.ndarray] = {}
        assigned_scores: dict[str, np.ndarray] = {}
        wrist_targets = {"left": body[7, :2], "right": body[4, :2]}
        for hand, scores in hand_candidates:
            point = hand[0, :2]
            distances = {
                side: _distance(point, wrist)
                for side, wrist in wrist_targets.items()
                if body[7 if side == "left" else 4, 2] > 0.05
            }
            side = min(distances, key=distances.get) if distances else (
                "left" if "left" not in assigned_hands else "right"
            )
            if side in assigned_hands:
                side = "right" if side == "left" else "left"
            assigned_hands[side] = hand
            assigned_scores[side] = scores

        people.append(
            _Person(
                body=body,
                face=face,
                hands=assigned_hands,
                body_confidence=body_confidence[person_index].copy(),
                face_confidence=face_confidence,
                hand_confidence=assigned_scores,
            )
        )
    return people


def _body_measurements(person: _Person) -> tuple[tuple[float, float], float]:
    body = person.body
    relevant = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13]
    valid = [
        index for index in relevant
        if body[index, 2] >= 0.20 and np.isfinite(body[index, :2]).all()
    ]
    if len(valid) < 4:
        return (0.5, 0.5), 0.0
    points = body[valid, :2]
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    hip_indices = [index for index in (8, 11) if body[index, 2] >= 0.20]
    if hip_indices:
        center = tuple(np.mean(body[hip_indices, :2], axis=0))
    else:
        center = (
            float((np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0),
            float((np.min(points[:, 1]) + np.max(points[:, 1])) / 2.0),
        )
    return (float(center[0]), float(center[1])), height


def _choose_person(people: list[_Person]) -> Optional[_Person]:
    best = None
    best_score = -1.0
    for person in people:
        valid = person.body_confidence > 0.20
        if int(valid.sum()) < 5:
            continue
        points = person.body[valid, :2]
        area = max(
            1e-5,
            float(
                (np.max(points[:, 0]) - np.min(points[:, 0])) *
                (np.max(points[:, 1]) - np.min(points[:, 1]))
            ),
        )
        score = float(np.mean(person.body_confidence[valid])) * math.sqrt(area)
        if score > best_score:
            best, best_score = person, score
    return best


def _ear(face: np.ndarray, indices: tuple[int, ...]) -> Optional[float]:
    if len(indices) != 6 or np.min(face[list(indices), 2]) < 0.20:
        return None
    points = face[:, :2]
    width = _distance(points[indices[0]], points[indices[3]])
    if width < 1e-6:
        return None
    return (
        _distance(points[indices[1]], points[indices[5]]) +
        _distance(points[indices[2]], points[indices[4]])
    ) / (2.0 * width)


def _mouth_ratios(
    face: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if np.min(face[[48, 51, 57, 54, 62, 66], 2]) < 0.20:
        return None, None, None
    points = face[:, :2]
    mouth_width = _distance(points[48], points[54])
    if mouth_width < 1e-6:
        return None, None, None
    jaw_open = _distance(points[62], points[66]) / mouth_width
    mouth_center_y = (points[51, 1] + points[57, 1]) / 2.0
    left_smile = (mouth_center_y - points[48, 1]) / mouth_width
    right_smile = (mouth_center_y - points[54, 1]) / mouth_width
    return jaw_open, left_smile, right_smile


def _body_bone_angles(
    person: _Person,
) -> tuple[dict[str, float], dict[str, float]]:
    body = person.body
    angles: dict[str, float] = {}
    confidence: dict[str, float] = {}
    for name, (start, end) in BODY_BONES.items():
        if min(body[start, 2], body[end, 2]) < 0.20:
            continue
        angles[name] = _angle_2d(body[start, :2], body[end, :2])
        confidence[name] = float(min(body[start, 2], body[end, 2]))
    for name, (a, b) in (("Chest", (2, 5)), ("Hips", (8, 11))):
        if min(body[a, 2], body[b, 2]) >= 0.20:
            angles[name] = _angle_2d(body[a, :2], body[b, :2])
            confidence[name] = float(min(body[a, 2], body[b, 2]))
    if body[1, 2] > 0.20 and min(body[8, 2], body[11, 2]) > 0.20:
        hip_center = (body[8, :2] + body[11, :2]) / 2.0
        angles["Spine"] = _angle_2d(hip_center, body[1, :2])
        confidence["Spine"] = float(min(body[1, 2], body[8, 2], body[11, 2]))
    return angles, confidence


def _hand_bone_angles(
    person: _Person,
) -> tuple[dict[str, float], dict[str, float]]:
    angles: dict[str, float] = {}
    confidence: dict[str, float] = {}
    for side, hand in person.hands.items():
        scores = person.hand_confidence[side]
        for finger, points, joints in HAND_FINGERS:
            for joint_index, joint_name in enumerate(joints):
                triplet = points[joint_index:joint_index + 3]
                if len(triplet) < 3 or min(scores[list(triplet)]) < 0.20:
                    continue
                key = f"{side.title()}{finger}{joint_name}"
                angles[key] = _joint_angle(
                    hand[triplet[0]], hand[triplet[1]], hand[triplet[2]]
                )
                confidence[key] = float(min(scores[list(triplet)]))
    return angles, confidence
