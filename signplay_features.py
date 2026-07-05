"""Shared landmark normalization for SignPlay ASL (60D feature vectors)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
NUM_LANDMARKS = 21
FEATURE_DIM = 60
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

HAND_CONNECTIONS = (
    vision.HandLandmarksConnections.HAND_PALM_CONNECTIONS
    + vision.HandLandmarksConnections.HAND_THUMB_CONNECTIONS
    + vision.HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS
    + vision.HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS
    + vision.HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS
    + vision.HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS
)


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to shape (21, 3)."""
    if hasattr(hand_landmarks, "landmark"):
        points = hand_landmarks.landmark
    else:
        points = hand_landmarks
    return np.array([[lm.x, lm.y, lm.z] for lm in points], dtype=np.float64)


def apply_mirror_symmetry(raw: np.ndarray) -> np.ndarray:
    """Mirror left-hand X coordinates (multiply x by -1) before normalization."""
    mirrored = raw.copy()
    mirrored[:, 0] *= -1.0
    return mirrored


def normalize_to_feature_vector(raw: np.ndarray) -> np.ndarray:
    """
    Translation: subtract wrist P0 from landmarks 1..20.
    Scale: divide by L_palm = ||P9 - P0||.
    Returns flattened 60D vector.
    """
    if raw.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected landmarks shape ({NUM_LANDMARKS}, 3), got {raw.shape}")

    wrist = raw[WRIST_IDX]
    palm_size = float(np.linalg.norm(raw[MIDDLE_MCP_IDX] - wrist))
    if palm_size < 1e-6:
        palm_size = 1e-6

    relative = (raw[1:] - wrist) / palm_size
    return relative.reshape(FEATURE_DIM).astype(np.float32)


class HandLandmarkDetector:
    """MediaPipe hand detector with safe fallbacks when hands leave the frame."""

    def __init__(self, camera_index: int = 0) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH.name}. Download MediaPipe hand_landmarker.task first."
            )

        self._camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._timestamp_ms = 0

    def open_camera(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._camera_index}")

    def read_frame(self) -> tuple[bool, np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Camera is not open")
        success, frame = self._cap.read()
        if not success:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, cv2.flip(frame, 1)

    def process(self, bgr_frame: np.ndarray) -> tuple[Optional[np.ndarray], Optional[list]]:
        """
        Returns (feature_vector_60d, hand_landmarks_list_for_drawing).
        Never raises when hand is missing.
        """
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

            if not result.hand_landmarks:
                return None, None

            landmarks = result.hand_landmarks[0]
            raw = landmarks_to_array(landmarks)

            if result.handedness:
                category = result.handedness[0][0].category_name
                if category == "Left":
                    raw = apply_mirror_symmetry(raw)

            feature = normalize_to_feature_vector(raw)
            return feature, landmarks
        except Exception:
            return None, None

    def draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        try:
            vision.drawing_utils.draw_landmarks(frame, landmarks, HAND_CONNECTIONS)
        except Exception:
            return

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._landmarker.close()
