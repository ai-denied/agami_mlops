"""
손전등 캡챠 이미지 데이터셋 로더
================================
captcha_labels/*.json 1000개를 앱 시작 시 1회 인덱싱.
챌린지 생성 시 무작위 3장을 중복 없이 선택.
bbox px 좌표를 정규화 좌표(0~1)로 변환해 보관.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_DIR = Path(__file__).resolve().parent.parent / "static" / "captcha_labels"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "static" / "captcha_images"
IMAGE_URL_PREFIX = "/static/captcha_images"
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

# target_object id → 사용자에게 보일 한국어 라벨.
# 모르는 id 는 dataset 로드 시점에 스킵 (안전장치).
TARGET_OBJECT_LABELS: dict[str, str] = {
    "cup1": "파란색 컵",
    "cup2": "회색 컵",
    "cup3": "작은 컵",
    "cup4": "파란색 컵",
    "cup5": "파란색 컵",
    "key1": "금색 열쇠",
    "key2": "노란색 열쇠",
    "key3": "노란색 열쇠",
    "key4": "은색 열쇠",
    "key5": "초록/금색 열쇠",
    "shoes1": "하얀색 운동화",
    "shoes2": "파란색 운동화",
    "shoes3": "베이지색 운동화",
    "shoes4": "알록달록한 운동화",
    "shoes5": "흰/검 운동화",
    "라이언": "라이언",
    "무지": "무지",
    "어피치": "어피치",
    "춘식이": "춘식이",
    "프로도": "프로도",
}


class CaptchaImageEntry:
    """단일 이미지 + 라벨. 좌표는 모두 0~1 정규화."""

    __slots__ = (
        "image_filename",
        "target_object_id",
        "target_label",
        "bbox_x_norm",
        "bbox_y_norm",
        "bbox_w_norm",
        "bbox_h_norm",
        "center_x_norm",
        "center_y_norm",
    )

    def __init__(self, image_filename: str, target_object_id: str, bbox: dict) -> None:
        self.image_filename = image_filename
        self.target_object_id = target_object_id
        self.target_label = TARGET_OBJECT_LABELS.get(target_object_id, target_object_id)
        # 픽셀 → 정규화 (0~1)
        self.bbox_x_norm = bbox["x"] / IMAGE_WIDTH
        self.bbox_y_norm = bbox["y"] / IMAGE_HEIGHT
        self.bbox_w_norm = bbox["width"] / IMAGE_WIDTH
        self.bbox_h_norm = bbox["height"] / IMAGE_HEIGHT
        # 정답 좌표 = bbox 중심
        self.center_x_norm = self.bbox_x_norm + self.bbox_w_norm / 2
        self.center_y_norm = self.bbox_y_norm + self.bbox_h_norm / 2

    def image_url(self) -> str:
        return f"{IMAGE_URL_PREFIX}/{self.image_filename}"


_DATASET: list[CaptchaImageEntry] | None = None


def _load_dataset() -> list[CaptchaImageEntry]:
    """앱 시작 시 1회. 모든 captcha_*.json 을 인덱싱."""
    entries: list[CaptchaImageEntry] = []
    skipped_unknown_target = 0
    skipped_missing_image = 0
    skipped_parse_error = 0

    for json_path in sorted(LABEL_DIR.glob("captcha_*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            image_filename = data["image"]
            if not (IMAGE_DIR / image_filename).exists():
                skipped_missing_image += 1
                continue
            target_id = data["target_object"]
            if target_id not in TARGET_OBJECT_LABELS:
                skipped_unknown_target += 1
                continue
            entries.append(
                CaptchaImageEntry(
                    image_filename=image_filename,
                    target_object_id=target_id,
                    bbox=data["bbox"],
                )
            )
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            skipped_parse_error += 1
            continue

    if len(entries) < 3:
        raise RuntimeError(
            f"이미지 데이터셋 로드 실패: {len(entries)}개만 로드됨 (3장 묶음 발급 불가)."
        )
    logger.info(
        "flashlight image dataset loaded: %d entries "
        "(skipped: unknown_target=%d missing_image=%d parse_error=%d)",
        len(entries),
        skipped_unknown_target,
        skipped_missing_image,
        skipped_parse_error,
    )
    return entries


def get_dataset() -> list[CaptchaImageEntry]:
    """lazy singleton. 첫 호출 시 인덱싱."""
    global _DATASET
    if _DATASET is None:
        _DATASET = _load_dataset()
    return _DATASET


def pick_random_entries(rng, k: int = 3) -> list[CaptchaImageEntry]:
    """k개의 서로 다른 이미지를 균등 무작위 선택."""
    dataset = get_dataset()
    if len(dataset) < k:
        raise RuntimeError(f"데이터셋이 {len(dataset)}개. {k}개 선택 불가.")
    return rng.sample(dataset, k=k)


__all__ = [
    "CaptchaImageEntry",
    "TARGET_OBJECT_LABELS",
    "IMAGE_URL_PREFIX",
    "IMAGE_WIDTH",
    "IMAGE_HEIGHT",
    "get_dataset",
    "pick_random_entries",
]
