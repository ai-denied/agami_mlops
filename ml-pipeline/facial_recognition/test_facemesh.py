import cv2
import mediapipe as mp
from pathlib import Path
import random

DATASET_DIR = Path("dataset/face_spoof")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


image_paths = []
for label in ["real", "spoof"]:
    image_paths.extend([
        p for p in (DATASET_DIR / label).rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ])

sample_paths = random.sample(image_paths, min(10, len(image_paths)))

mp_face_mesh = mp.solutions.face_mesh

success = 0
fail = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    for path in sample_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"[READ_FAIL] {path}")
            fail += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            print(f"[OK] {path} / landmarks: {len(landmarks)}")
            success += 1
        else:
            print(f"[FAIL] {path}")
            fail += 1

print("\n=== FaceMesh Test Result ===")
print(f"Total: {len(sample_paths)}")
print(f"Success: {success}")
print(f"Fail: {fail}")
print(f"Success Rate: {success / len(sample_paths) * 100:.2f}%")