"""Calibration, expression mapping, and tracking-loss recovery for skeletal frames."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .skeletal_frame import (
    BODY_LANDMARK_NAMES,
    HAND_FINGERS,
    SkeletalFrame,
    _body_bone_angles,
    _body_measurements,
    _choose_person,
    _clamp,
    _distance,
    _ear,
    _finite,
    _hand_bone_angles,
    _mouth_ratios,
    _pose_people,
    _wrap_angle,
)


def _deadband(delta: float, threshold: float) -> float:
    """Remove small frame-to-frame detector noise without freezing larger motion."""
    if abs(delta) <= threshold:
        return 0.0
    return delta - threshold if delta > 0 else delta + threshold


class SkeletalPoseMapper:
    """Maps DWPose keypoints to calibrated, smoothed body-height-relative frames."""

    def __init__(self, smoothing: float = 0.45, loss_recovery: float = 0.18):
        self.smoothing = _clamp(smoothing, 0.05, 1.0)
        self.loss_recovery = _clamp(loss_recovery, 0.01, 1.0)
        self._anchor_center: Optional[tuple[float, float]] = None
        self._anchor_height: Optional[float] = None
        self._neutral_bones: dict[str, float] = {}
        self._neutral_eye = {"left": 0.28, "right": 0.28}
        self._neutral_mouth = 0.025
        self._neutral_smile = {"left": 0.0, "right": 0.0}
        self._rest_landmarks: dict[str, tuple[float, float, float]] = {}
        self._smoothed_landmarks: dict[str, tuple[float, float, float]] = {}
        self._smoothed_image_landmarks: dict[str, tuple[float, float]] = {}
        self._smoothed_bones: dict[str, tuple[float, float, float]] = {}
        self._smoothed_blendshapes: dict[str, float] = {}
        self._previous_root = (0.0, 0.0, 0.0)
        self._previous_timestamp_ms: Optional[int] = None
        self._previous_ankles: dict[str, tuple[float, float]] = {}
        self._locked_feet: dict[str, tuple[float, float]] = {}

    @property
    def calibrated(self) -> bool:
        return self._anchor_center is not None and self._anchor_height is not None

    def calibrate(self, pose: dict) -> None:
        person = _choose_person(_pose_people(pose))
        if person is None:
            raise ValueError("No reliable body pose is visible for calibration.")
        center, height = _body_measurements(person)
        if height < 0.15:
            raise ValueError("Show more of your body before calibrating.")
        self._anchor_center = center
        self._anchor_height = height
        self._neutral_bones, _ = _body_bone_angles(person)
        hand_angles, _ = _hand_bone_angles(person)
        self._neutral_bones.update(hand_angles)
        left_eye = _ear(person.face, (36, 37, 38, 39, 40, 41))
        right_eye = _ear(person.face, (42, 43, 44, 45, 46, 47))
        if left_eye is not None:
            self._neutral_eye["left"] = max(0.08, left_eye)
        if right_eye is not None:
            self._neutral_eye["right"] = max(0.08, right_eye)
        jaw, left_smile, right_smile = _mouth_ratios(person.face)
        if jaw is not None:
            self._neutral_mouth = max(0.0, jaw)
        if left_smile is not None:
            self._neutral_smile["left"] = left_smile
        if right_smile is not None:
            self._neutral_smile["right"] = right_smile
        self._rest_landmarks = self._to_landmarks(person, center, height, 0.0)
        self._locked_feet.clear()
        self._previous_ankles.clear()
        self._smoothed_landmarks = dict(self._rest_landmarks)
        self._smoothed_image_landmarks = {}
        self._smoothed_bones = {}
        self._smoothed_blendshapes = {}
        self._previous_root = (0.0, 0.0, 0.0)

    def _to_landmarks(
        self,
        person,
        center: tuple[float, float],
        height: float,
        depth: float,
    ) -> dict[str, tuple[float, float, float]]:
        if height <= 1e-6:
            return {}
        landmarks: dict[str, tuple[float, float, float]] = {}

        def add(name: str, point: np.ndarray):
            if point[2] < 0.05:
                return
            x = (float(point[0]) - center[0]) / height
            y = (center[1] - float(point[1])) / height
            landmarks[name] = (_finite(x), _finite(y), _finite(depth))

        for index, name in enumerate(BODY_LANDMARK_NAMES):
            add(name, person.body[index])
        if np.max(person.face[:, 2]) > 0.05:
            for index, point in enumerate(person.face):
                add(f"face_{index:02d}", point)
        for side, hand in person.hands.items():
            for index, point in enumerate(hand):
                if person.hand_confidence[side][index] >= 0.05:
                    add(f"{side}_hand_{index:02d}", point)
        return landmarks

    def _image_landmarks(
        self, person
    ) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
        points: dict[str, tuple[float, float]] = {}
        confidence: dict[str, float] = {}

        def add(name: str, point: np.ndarray):
            score = _clamp(point[2], 0.0, 1.0)
            confidence[name] = score
            if score >= 0.05:
                points[name] = (
                    _clamp(point[0], 0.0, 1.0),
                    _clamp(point[1], 0.0, 1.0),
                )

        for index, name in enumerate(BODY_LANDMARK_NAMES):
            add(name, person.body[index])
        for index, point in enumerate(person.face):
            add(f"face_{index:02d}", point)
        for side, hand in person.hands.items():
            scores = person.hand_confidence[side]
            for index, point in enumerate(hand):
                add(
                    f"{side}_hand_{index:02d}",
                    np.array([point[0], point[1], scores[index]]),
                )
        return points, confidence

    def _expressions(self, person) -> tuple[dict[str, float], dict[str, float]]:
        face = person.face
        left_eye = _ear(face, (36, 37, 38, 39, 40, 41))
        right_eye = _ear(face, (42, 43, 44, 45, 46, 47))
        jaw, left_smile, right_smile = _mouth_ratios(face)
        raw = {
            "eyeBlinkLeft": (
                _clamp(
                    (self._neutral_eye["left"] - left_eye) /
                    max(0.08, self._neutral_eye["left"] * 0.55),
                    0.0, 1.0,
                ) if left_eye is not None else 0.0
            ),
            "eyeBlinkRight": (
                _clamp(
                    (self._neutral_eye["right"] - right_eye) /
                    max(0.08, self._neutral_eye["right"] * 0.55),
                    0.0, 1.0,
                ) if right_eye is not None else 0.0
            ),
            "jawOpen": (
                _clamp((jaw - self._neutral_mouth) / 0.28, 0.0, 1.0)
                if jaw is not None else 0.0
            ),
            "mouthSmileLeft": (
                _clamp(
                    (left_smile - self._neutral_smile["left"]) / 0.15,
                    0.0, 1.0,
                ) if left_smile is not None else 0.0
            ),
            "mouthSmileRight": (
                _clamp(
                    (right_smile - self._neutral_smile["right"]) / 0.15,
                    0.0, 1.0,
                ) if right_smile is not None else 0.0
            ),
        }
        face_quality = (
            float(np.mean(person.face_confidence))
            if len(person.face_confidence) else 0.0
        )
        confidences = {key: _clamp(face_quality, 0.0, 1.0) for key in raw}
        return raw, confidences

    def _update_foot_contacts(
        self, person, timestamp_ms: int
    ) -> tuple[dict[str, bool], dict[str, tuple[float, float]]]:
        contacts = {"left": False, "right": False}
        adjusted: dict[str, tuple[float, float]] = {}
        dt = (
            max(0.001, (timestamp_ms - self._previous_timestamp_ms) / 1000.0)
            if self._previous_timestamp_ms is not None else 1.0 / 30.0
        )
        for side, index in (("right", 10), ("left", 13)):
            point = person.body[index]
            if point[2] < 0.35:
                self._locked_feet.pop(side, None)
                continue
            current = (float(point[0]), float(point[1]))
            previous = self._previous_ankles.get(side)
            speed = (
                _distance(current, previous) /
                max(self._anchor_height or 0.5, 1e-3) / dt
                if previous is not None else 0.0
            )
            if side in self._locked_feet and speed > 0.40:
                self._locked_feet.pop(side, None)
            elif (
                side not in self._locked_feet
                and current[1] > 0.68
                and speed < 0.18
            ):
                self._locked_feet[side] = current
            self._previous_ankles[side] = current
            if side in self._locked_feet:
                contacts[side] = True
                adjusted[side] = self._locked_feet[side]
        return contacts, adjusted

    def update(
        self, pose: dict, timestamp_ms: int, image_shape: tuple[int, ...]
    ) -> SkeletalFrame:
        person = _choose_person(_pose_people(pose))
        if person is None:
            return self._lost_frame(timestamp_ms)

        center, body_height = _body_measurements(person)
        if body_height < 0.05:
            return self._lost_frame(timestamp_ms)
        if self.calibrated:
            anchor_center = self._anchor_center
            anchor_height = max(float(self._anchor_height), 1e-6)
            root_z = _clamp(
                anchor_height / max(body_height, 1e-6) - 1.0, -0.65, 0.85
            )
        else:
            anchor_center = (0.5, 0.5)
            anchor_height = max(body_height, 0.2)
            root_z = 0.0

        root = (
            _finite((center[0] - anchor_center[0]) / anchor_height),
            _finite((anchor_center[1] - center[1]) / anchor_height),
            _finite(root_z),
        )
        contacts, foot_adjustments = self._update_foot_contacts(person, timestamp_ms)
        image_landmarks, confidence = self._image_landmarks(person)
        for side, point in foot_adjustments.items():
            image_landmarks[f"{side}_ankle"] = point

        spatial = self._to_landmarks(person, center, body_height, 0.0)
        for side, point in foot_adjustments.items():
            spatial[f"{side}_ankle"] = (
                (point[0] - center[0]) / body_height,
                (center[1] - point[1]) / body_height,
                0.0,
            )

        raw_angles, angle_confidence = _body_bone_angles(person)
        hand_angles, hand_confidence = _hand_bone_angles(person)
        raw_angles.update(hand_angles)
        angle_confidence.update(hand_confidence)
        bones = {}
        for name, angle in raw_angles.items():
            neutral = self._neutral_bones.get(name, angle) if self.calibrated else angle
            delta = _wrap_angle(angle - neutral)
            is_hand = any(finger in name for finger, _, _ in HAND_FINGERS)
            axis = (delta, 0.0, 0.0) if is_hand else (0.0, 0.0, delta)
            previous = self._smoothed_bones.get(name, (0.0, 0.0, 0.0))
            bones[name] = tuple(
                _finite(
                    previous[index] +
                    self.smoothing * _deadband(
                        axis[index] - previous[index], 0.035
                    )
                )
                for index in range(3)
            )
        for name, previous in list(self._smoothed_bones.items()):
            if name not in bones:
                bones[name] = tuple(
                    _finite(value * (1.0 - self.loss_recovery))
                    for value in previous
                )
        self._smoothed_bones = bones

        raw_blendshapes, blend_confidence = self._expressions(person)
        blendshapes = {}
        for name, value in raw_blendshapes.items():
            previous = self._smoothed_blendshapes.get(name, 0.0)
            blendshapes[name] = _clamp(
                previous + self.smoothing * (value - previous), 0.0, 1.0
            )
        self._smoothed_blendshapes = blendshapes

        landmarks = self._smooth_landmarks(spatial, confidence)
        image_landmarks = self._smooth_image_landmarks(
            image_landmarks, confidence
        )
        root_delta = tuple(
            _deadband(root[index] - self._previous_root[index], 0.02)
            for index in range(3)
        )
        self._previous_root = tuple(
            self._previous_root[index] +
            self.smoothing * root_delta[index]
            for index in range(3)
        )
        self._previous_timestamp_ms = timestamp_ms
        return SkeletalFrame(
            timestamp_ms=int(timestamp_ms),
            tracked=True,
            calibrated=self.calibrated,
            root_translation=tuple(_finite(value) for value in self._previous_root),
            landmarks=landmarks,
            image_landmarks=image_landmarks,
            bones=bones,
            blendshapes=blendshapes,
            confidence={
                **confidence,
                **angle_confidence,
                **blend_confidence,
            },
            foot_contacts=contacts,
            depth_mode=(
                "monocular_body_scale_estimate" if self.calibrated else "uncalibrated"
            ),
        )

    def _smooth_landmarks(
        self,
        current: dict[str, tuple[float, float, float]],
        confidence: dict[str, float],
    ) -> dict[str, tuple[float, float, float]]:
        result = {}
        keys = set(current) | set(self._smoothed_landmarks) | set(self._rest_landmarks)
        for name in keys:
            if name in current and confidence.get(name, 0.0) >= 0.20:
                target = current[name]
                previous = self._smoothed_landmarks.get(name, target)
                value = tuple(
                    previous[index] +
                    self.smoothing * _deadband(
                        target[index] - previous[index], 0.012
                    )
                    for index in range(3)
                )
            else:
                previous = self._smoothed_landmarks.get(
                    name, self._rest_landmarks.get(name, (0.0, 0.0, 0.0))
                )
                target = self._rest_landmarks.get(name, (0.0, 0.0, 0.0))
                value = tuple(
                    previous[index] +
                    self.loss_recovery * (target[index] - previous[index])
                    for index in range(3)
                )
            result[name] = tuple(_finite(component) for component in value)
        self._smoothed_landmarks = result
        return result

    def _smooth_image_landmarks(
        self,
        current: dict[str, tuple[float, float]],
        confidence: dict[str, float],
    ) -> dict[str, tuple[float, float]]:
        result = {}
        for name, target in current.items():
            if confidence.get(name, 0.0) < 0.20:
                continue
            previous = self._smoothed_image_landmarks.get(name, target)
            result[name] = tuple(
                _finite(
                    previous[index] +
                    self.smoothing * _deadband(
                        target[index] - previous[index], 0.004
                    )
                )
                for index in range(2)
            )
        self._smoothed_image_landmarks = result
        return result

    def _lost_frame(self, timestamp_ms: int) -> SkeletalFrame:
        root = tuple(
            _finite(value * (1.0 - self.loss_recovery))
            for value in self._previous_root
        )
        self._previous_root = root
        landmarks = self._smooth_landmarks({}, {})
        bones = {
            name: tuple(
                _finite(value * (1.0 - self.loss_recovery)) for value in values
            )
            for name, values in self._smoothed_bones.items()
        }
        self._smoothed_bones = bones
        blendshapes = {
            name: _clamp(value * (1.0 - self.loss_recovery), 0.0, 1.0)
            for name, value in self._smoothed_blendshapes.items()
        }
        self._smoothed_blendshapes = blendshapes
        self._previous_timestamp_ms = timestamp_ms
        return SkeletalFrame(
            timestamp_ms=int(timestamp_ms),
            tracked=False,
            calibrated=self.calibrated,
            root_translation=root,
            landmarks=landmarks,
            bones=bones,
            blendshapes=blendshapes,
            confidence={},
            foot_contacts={"left": False, "right": False},
            depth_mode=(
                "monocular_body_scale_estimate" if self.calibrated else "uncalibrated"
            ),
        )
