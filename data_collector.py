"""
SignPlay ASL — interactive landmark data collector.

Press S to save the current 60D normalized vector + label to dataset.csv.
Press Q to quit.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2

from signplay_features import HandLandmarkDetector

DATASET_PATH = Path(__file__).with_name("dataset.csv")
VALID_LABELS = {*(chr(code) for code in range(ord("A"), ord("Z") + 1)), *(str(d) for d in range(10))}


def prompt_target_class() -> str:
    while True:
        label = input("Enter target class (A-Z or 0-9): ").strip().upper()
        if label in VALID_LABELS:
            return label
        print(f"Invalid label '{label}'. Use A-Z or 0-9.")


def load_class_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    if not DATASET_PATH.exists():
        return counts

    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[row["label"]] += 1
    return counts


def append_sample(label: str, feature_vector) -> None:
    header = ["label", *[f"f{i}" for i in range(60)]]
    file_exists = DATASET_PATH.exists()

    with DATASET_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([label, *feature_vector.tolist()])


def draw_ui(frame, label: str, class_count: int, total_saved: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (620, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Target class: {label}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Saved this class: {class_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Total saved: {total_saved}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "S = save sample | Q = quit", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def run_collector() -> None:
    label = prompt_target_class()
    counts = load_class_counts()
    detector = HandLandmarkDetector()

    try:
        detector.open_camera()
        print(f"Collecting samples for class '{label}'. Press S to save, Q to quit.")

        while True:
            ok, frame = detector.read_frame()
            if not ok:
                continue

            feature, landmarks = detector.process(frame)
            if landmarks is not None:
                detector.draw_landmarks(frame, landmarks)

            draw_ui(frame, label, counts[label], sum(counts.values()))
            cv2.imshow("SignPlay Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                if feature is None:
                    print("No hand detected — sample not saved.")
                    continue
                append_sample(label, feature)
                counts[label] += 1
                print(f"Saved sample #{counts[label]} for class '{label}'.")
    finally:
        detector.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_collector()
