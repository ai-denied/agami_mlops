"""
캡챠 챌린지 데이터 모델
=======================
WBS #41: 캡챠 문제(Challenge) 유형 정의

- 모든 캡챠 종류(손전등/안면/맥락추론)를 포괄하는 공용 베이스 + 종류별 스펙
- ChallengeSpec: 클라이언트로 내려가는 정보 (정답 좌표 제외)
- ChallengeAnswer: 서버 내부 보관용 (Redis 저장 대상, 클라이언트 노출 금지)

Pydantic v2 기준. FastAPI에서 그대로 response_model 로 사용 가능.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 공통 enum
# ---------------------------------------------------------------------------

class ChallengeKind(str, Enum):
    """캡챠 종류 (기획서 2.1 핵심 캡챠 서비스 3종)"""
    FLASHLIGHT = "flashlight"            # 2.1.1 암흑 속 손전등 드래그
    FACE_MISSION = "face_mission"        # 2.1.2 실시간 안면 미션 (추후 구현)
    CONTEXT_INFERENCE = "context_inference"  # 2.1.3 이미지 상황 맥락 추론 (추후 구현)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# 손전등 캡챠 (flashlight) - 변형(variant)
# ---------------------------------------------------------------------------

class FlashlightVariant(str, Enum):
    """
    손전등 캡챠의 변형. 같은 손전등 종류 안에서도 문제 유형을 다양화하기 위함.
    - SINGLE_TARGET: 화면에 정답 객체 1개만 존재 (가장 쉬움, easy 기본)
    - AMONG_DECOYS:  정답 1개 + 비슷한 미끼 N개 (medium/hard)
    - MULTI_TARGET:  같은 종류의 객체가 여러 개 보이고, 그중 특정 색/모양만 정답
                     (확장용, 본 PR에서는 스키마만 정의)
    """
    SINGLE_TARGET = "single_target"
    AMONG_DECOYS = "among_decoys"
    MULTI_TARGET = "multi_target"


class FlashlightObject(BaseModel):
    """
    화면에 등장하는 객체 1개.
    좌표는 0.0 ~ 1.0 비율. 캔버스 픽셀 크기와 독립적이므로 반응형에서도 안전.
    """
    object_id: str = Field(..., description="객체 식별자 (예: 'key', 'coin')")
    label: str = Field(..., description="사용자에게 보일 한국어 이름 (예: '열쇠')")
    emoji: str = Field(..., description="이모지 표현 (예: '🔑')")
    x: Annotated[float, Field(ge=0.0, le=1.0)]
    y: Annotated[float, Field(ge=0.0, le=1.0)]


class FlashlightTargetHint(BaseModel):
    """
    클라이언트에게 '무엇을 찾아야 하는지'만 알려줌.
    좌표(x, y)는 일부러 포함하지 않음 -> 정답 노출 방지.
    """
    object_id: str
    label: str
    emoji: str


# ---------------------------------------------------------------------------
# Spec: 클라이언트로 내려가는 챌린지 사양
# ---------------------------------------------------------------------------

class ChallengeSpecBase(BaseModel):
    """모든 챌린지 spec 의 공통 필드."""
    challenge_id: str = Field(..., description="HMAC 또는 token_urlsafe 기반 1회용 ID")
    kind: ChallengeKind
    difficulty: Difficulty
    issued_at: datetime
    expires_at: datetime


class FlashlightSubChallenge(BaseModel):
    """
    번들 내 1장 sub-challenge.
    1챌린지 = 3장 묶음에서 각 그림 단위.
    """
    index: int = Field(..., ge=0, le=2, description="0-based 순서 (0,1,2).")
    image_url: str = Field(
        ...,
        description="배경 이미지 URL (예: /static/captcha_images/captcha_0001.jpg).",
    )
    target_hint: FlashlightTargetHint
    decoys: list[FlashlightObject] = Field(
        default_factory=list,
        description="이 그림에 함께 그려질 미끼 객체들 (이미지 데이터셋 사용 시 빈 배열).",
    )


class FlashlightChallengeSpec(ChallengeSpecBase):
    """손전등 캡챠 1회 인스턴스의 클라이언트 측 사양 (1챌린지 = 3장 묶음)."""
    kind: Literal[ChallengeKind.FLASHLIGHT] = ChallengeKind.FLASHLIGHT
    variant: FlashlightVariant

    sub_challenges: list[FlashlightSubChallenge] = Field(
        ..., min_length=3, max_length=3,
        description="3장의 sub-challenge. 사용자는 순서대로 1→2→3 진행.",
    )

    flashlight_radius: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        ..., description="손전등이 비추는 반경 (캔버스 짧은 변 기준 비율). 3장 공유."
    )
    time_limit_sec: Annotated[int, Field(gt=0, le=300)] = Field(
        ..., description="3장 전체에 적용되는 총 시간 (연속 카운트다운)."
    )
    hint_after_sec: int | None = Field(
        None, description="N초 경과 시 힌트 표시. None 이면 힌트 없음. 번들 단위."
    )
    canvas_aspect_w: int = 16
    canvas_aspect_h: int = 9


# ---------------------------------------------------------------------------
# Answer: 서버 내부 보관용 (Redis 저장)
# ---------------------------------------------------------------------------

class FlashlightSubAnswer(BaseModel):
    """번들 내 1장 sub-answer.

    bbox_w/bbox_h가 둘 다 0보다 크면 verifier가 bbox 사각형 매칭을 사용하고,
    그렇지 않으면 tolerance 기반 원형 매칭으로 fallback 한다.

    image_url / target_label 은 검증에는 쓰이지 않고 로컬 로깅(captcha_logger)
    에서 분석용으로 참조하기 위해 함께 보관한다.
    """
    index: int = Field(..., ge=0, le=2)
    correct_object_id: str
    correct_x: Annotated[float, Field(ge=0.0, le=1.0)]
    correct_y: Annotated[float, Field(ge=0.0, le=1.0)]
    tolerance: Annotated[float, Field(gt=0.0, le=0.2)] = Field(
        default=0.1,
        description="원형 매칭 fallback 반경 (bbox가 0일 때만 사용).",
    )
    bbox_w: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="정답 bbox 정규화 폭. 0이면 tolerance 원형 매칭으로 fallback.",
    )
    bbox_h: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="정답 bbox 정규화 높이.",
    )
    image_url: str = Field(default="", description="로그용 메타. 검증에는 미사용.")
    target_label: str = Field(default="", description="로그용 메타. 검증에는 미사용.")


class FlashlightChallengeAnswer(BaseModel):
    """
    서버 측에 보관되는 정답 (1챌린지 = 3장 묶음).
    절대 클라이언트로 직렬화하지 말 것.
    Redis key 권장: f"captcha:answer:{challenge_id}"
    Redis TTL 권장: time_limit_sec + 10  (네트워크 지연 마진)

    difficulty / flashlight_radius / time_limit_sec / canvas_aspect_*는
    검증 흐름과 무관한 메타로, 로컬 로깅(captcha_logger)이 분석할 때
    참고하기 위해 함께 보관한다.
    """
    challenge_id: str
    kind: Literal[ChallengeKind.FLASHLIGHT] = ChallengeKind.FLASHLIGHT
    sub_answers: list[FlashlightSubAnswer] = Field(
        ..., min_length=3, max_length=3,
        description="3장의 sub-answer. spec.sub_challenges와 index로 매칭.",
    )
    created_at: datetime
    expires_at: datetime
    difficulty: str = Field(default="", description="로그용 메타.")
    flashlight_radius: float = Field(default=0.0, description="로그용 메타.")
    time_limit_sec: int = Field(default=0, description="로그용 메타.")
    canvas_aspect_w: int = Field(default=0, description="로그용 메타.")
    canvas_aspect_h: int = Field(default=0, description="로그용 메타.")


# ---------------------------------------------------------------------------
# 안면 미션 캡챠 (face_mission) - 기획서 2.1.2
# ---------------------------------------------------------------------------

class FaceInstructionType(str, Enum):
    """
    안면 미션 1회에 사용자가 수행할 동작 카테고리.
    팀원 MediaPipe 합류 시 각 type 별로 자동 감지 모듈 매핑.
    """
    BLINK_LEFT = "blink_left"
    BLINK_RIGHT = "blink_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SMILE = "smile"
    NOD = "nod"


# 사용자에게 보일 한국어 라벨. generator 가 spec 으로 바로 내려보냄.
FACE_INSTRUCTION_LABELS: dict[FaceInstructionType, str] = {
    FaceInstructionType.BLINK_LEFT: "왼쪽 눈 감기",
    FaceInstructionType.BLINK_RIGHT: "오른쪽 눈 감기",
    FaceInstructionType.TURN_LEFT: "고개 왼쪽",
    FaceInstructionType.TURN_RIGHT: "고개 오른쪽",
    FaceInstructionType.SMILE: "미소 짓기",
    FaceInstructionType.NOD: "고개 끄덕이기",
}


class FaceInstruction(BaseModel):
    """
    한 회차의 단일 지시.
    duration_sec 는 클라이언트 위젯이 다음 지시로 넘어가기 전 표시할 최소 시간.
    """
    type: FaceInstructionType
    label: str
    duration_sec: int = Field(..., gt=0, le=10)


class FaceChallengeSpec(ChallengeSpecBase):
    """안면 미션 캡챠 1회 인스턴스의 클라이언트 측 사양."""
    kind: Literal[ChallengeKind.FACE_MISSION] = ChallengeKind.FACE_MISSION

    instructions: list[FaceInstruction] = Field(
        ..., description="사용자가 순서대로 수행해야 할 지시 목록. 보통 1~3개."
    )
    time_limit_sec: Annotated[int, Field(gt=0, le=120)]
    hint_after_sec: int | None = Field(
        None, description="N초 경과 시 힌트 표시. None 이면 힌트 없음."
    )


class FaceChallengeAnswer(BaseModel):
    """
    서버 측에 보관되는 안면 미션 정답.
    expected_instruction_types 는 클라이언트가 제출한 completed_instructions 와 비교됨.
    """
    challenge_id: str
    kind: Literal[ChallengeKind.FACE_MISSION] = ChallengeKind.FACE_MISSION
    expected_instruction_types: list[str] = Field(
        ..., description="generator 가 만든 지시 타입의 순서. 제출 시 일치 여부로 1차 판정."
    )
    tolerance_sec: float = Field(
        default=1.0, gt=0.0, le=10.0,
        description="각 지시 수행 시간 허용 오차 (MediaPipe 합류 후 실제 사용)."
    )
    created_at: datetime
    expires_at: datetime


# ---------------------------------------------------------------------------
# 감정 맥락 추론 캡챠 (context_inference) - 기획서 2.1.3
# ---------------------------------------------------------------------------

class Emotion(str, Enum):
    """4지선다로 노출되는 감정 카테고리."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"


class ContextQuestion(BaseModel):
    """
    감정 맥락 추론 캡챠 1챌린지 안의 단일 문제.
    매 문제마다 독립적으로 셔플된 4지선다 choices 를 보유.
    """
    index: int = Field(..., ge=0, description="0-based 출제 순서.")
    image_url: str = Field(..., description="문제 이미지 URL.")
    choices: list[Emotion] = Field(
        ..., min_length=2, max_length=8,
        description="이 문제의 4지선다 보기 (정답 포함, 매 문제 독립 셔플)."
    )


class ContextChallengeSpec(ChallengeSpecBase):
    """
    감정 맥락 추론 캡챠 1회 인스턴스의 클라이언트 측 사양.
    1챌린지 = N개 문제 시퀀스. 클라이언트는 모든 문제를 순서대로 풀고
    submitted_answers 리스트로 한 번에 제출.
    """
    kind: Literal[ChallengeKind.CONTEXT_INFERENCE] = ChallengeKind.CONTEXT_INFERENCE

    questions: list[ContextQuestion] = Field(
        ..., min_length=1, max_length=8,
        description="출제 문제 시퀀스. 난이도별 2~4개."
    )
    total_count: int = Field(
        ..., gt=0, le=8,
        description="총 문제 수 (len(questions) 와 동일, 편의용 중복 필드)."
    )
    time_limit_sec: Annotated[int, Field(gt=0, le=120)]
    hint_after_sec: int | None = Field(
        None, description="N초 경과 시 힌트 표시. None 이면 힌트 없음."
    )


class ContextChallengeAnswer(BaseModel):
    """
    서버 측에 보관되는 감정 맥락 추론 정답.
    correct_answers 는 questions 의 index 순서대로 정답 감정 문자열을 나열.
    클라이언트의 submitted_answers 와 전체 일치해야 hit.
    """
    challenge_id: str
    kind: Literal[ChallengeKind.CONTEXT_INFERENCE] = ChallengeKind.CONTEXT_INFERENCE
    correct_answers: list[str] = Field(
        ..., min_length=1, max_length=8,
        description="출제 순서대로의 정답 감정 (Emotion enum 의 value)."
    )
    created_at: datetime
    expires_at: datetime
