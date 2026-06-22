"""Defensive label-string normalization for the v1 training dataset build.

captcha_bank_human_reviewed.csv / manual_images_labeled.csv already store
clean canonical class names (they were produced by emotion_mapping.py), so
in practice this layer fixes nothing on the current dataset - it exists so
that case differences or known typos (doubt_confusion / doubt_confusning /
"Doubt/Confusion", stray whitespace, etc.) in any future or external input
don't silently produce a new, wrong class instead of being caught by the
schema check in build_train_dataset_v1.py.
"""
import re

from emotion_mapping import EMOTION_CLASSES, SITUATION_CLASSES

_EMOTION_LOOKUP = {c: c for c in EMOTION_CLASSES}
_SITUATION_LOOKUP = {c: c for c in SITUATION_CLASSES}

# raw/typo variant -> canonical class. Keys are already lowercased and
# stripped of non-alnum separators by _normalize_key before lookup.
_ALIASES = {
    "doubtconfusion": "confusion",
    "doubtconfusning": "confusion",
    "doubtconfusing": "confusion",
    "engagement": None,   # explicitly out-of-schema, not a typo
    "surprise": None,     # explicitly out-of-schema, not a typo
    "test": None,         # stray placeholder value, not a real label
    "n/a": None,
    "none": None,
    "": None,
}


def _normalize_key(value):
    if value is None:
        return ""
    key = value.strip().lower()
    key = re.sub(r"[\s/_\-]+", "", key)
    return key


def normalize_emotion(value):
    """Returns a canonical EMOTION_CLASSES member, or None if unmapped/empty."""
    if not value:
        return None
    key = _normalize_key(value)
    if key in {_normalize_key(c) for c in EMOTION_CLASSES}:
        for c in EMOTION_CLASSES:
            if _normalize_key(c) == key:
                return c
    if key in _ALIASES:
        return _ALIASES[key]
    return "__unknown__"  # caller treats this as a schema violation


def normalize_situation(value):
    """Returns a canonical SITUATION_CLASSES member, or None if unmapped/empty."""
    if not value:
        return None
    key = _normalize_key(value)
    for c in SITUATION_CLASSES:
        if _normalize_key(c) == key:
            return c
    if key in _ALIASES:
        return _ALIASES[key]
    return "__unknown__"
