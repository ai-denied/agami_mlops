import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math

DATASET_DIR = Path("dataset/face_spoof")
OUT_DIR = Path("features")
OUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

LABEL_MAP = {
    "real": 0,
    "spoof": 1,
}

# FaceMesh 주요 landmark index
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_MOUTH = 61
RIGHT_MOUTH = 291
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def safe_ratio(num, den):
    return num / den if den != 0 else 0


def extract_features(landmarks):
    left_eye_open = dist(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM])
    right_eye_open = dist(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM])
    eye_open_ratio = (left_eye_open + right_eye_open) / 2

    mouth_open = dist(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
    mouth_width = dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    mouth_open_ratio = safe_ratio(mouth_open, mouth_width)

    smile_ratio = mouth_width

    face_height = dist(landmarks[FOREHEAD], landmarks[CHIN])
    face_width = dist(landmarks[234], landmarks[454])
    face_ratio = safe_ratio(face_width, face_height)

    nose_x = landmarks[NOSE_TIP].x
    nose_y = landmarks[NOSE_TIP].y

    return {
        "eye_open_ratio": eye_open_ratio,
        "mouth_open_ratio": mouth_open_ratio,
        "smile_ratio": smile_ratio,
        "face_width_height_ratio": face_ratio,
        "nose_x": nose_x,
        "nose_y": nose_y,
    }


image_paths = []
for label_name in ["real", "spoof"]:
    label_dir = DATASET_DIR / label_name
    for p in label_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            image_paths.append((p, label_name, LABEL_MAP[label_name]))

rows = []
fail_rows = []

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    for path, label_name, label_id in tqdm(image_paths, desc="Extracting FaceMesh features"):
        image = cv2.imread(str(path))

        if image is None:
            fail_rows.append({
                "file_path": str(path),
                "label": label_name,
                "reason": "read_fail"
            })
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            fail_rows.append({
                "file_path": str(path),
                "label": label_name,
                "reason": "no_face_detected"
            })
            continue

        landmarks = result.multi_face_landmarks[0].landmark
        features = extract_features(landmarks)

        row = {
            "file_path": str(path),
            "label": label_name,
            "label_id": label_id,
            **features
        }

        rows.append(row)

df = pd.DataFrame(rows)
fail_df = pd.DataFrame(fail_rows)

df.to_csv(OUT_DIR / "face_spoof_features.csv", index=False)
fail_df.to_csv(OUT_DIR / "face_spoof_failed.csv", index=False)

total = len(image_paths)
success = len(rows)
fail = len(fail_rows)

print("\n=== Face Spoof Feature Extraction Result ===")
print(f"Total images: {total}")
print(f"Success: {success}")
print(f"Fail: {fail}")
print(f"Success rate: {success / total * 100:.2f}%")
print(f"Saved: {OUT_DIR / 'face_spoof_features.csv'}")
print(f"Failed saved: {OUT_DIR / 'face_spoof_failed.csv'}")