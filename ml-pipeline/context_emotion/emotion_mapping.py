"""Final label scheme for the context_emotion CAPTCHA bank.

This is a RECONSTRUCTION of the mapping described in
retrospectives/2026-06-19 and 2026-06-21 (Day 4 / Day 5), after the
original work (review_app.py, restore_original_emotic_labels.py,
captcha_bank_human_reviewed.csv, qwen_attack_results.csv, ...) was lost
when the pod was reset. See RECONSTRUCTION_NOTES.md in this directory
for exactly what could and could not be reproduced.

Final classes - FIXED as of 2026-06-22 (see RECONSTRUCTION_NOTES.md
"2026-06-22 스키마 고정" / "everyday 제거" sections for the full history -
empathy was dropped, everyday was briefly reinstated then removed again
since no data source ever populates it):
  emotion 14:    happiness, calm, anticipation, affection, anger, fear,
                 sadness, disconnection, suffering, aversion,
                 embarrassment, confidence, confusion, yearning
  situation 7:   conflict, danger, loss_absence, pressure, safety,
                 teasing, vanity
"""

EMOTION_CLASSES = [
    "happiness", "calm", "anticipation", "affection", "anger", "fear",
    "sadness", "disconnection", "suffering", "aversion", "embarrassment",
    "confidence", "confusion", "yearning",
]

SITUATION_CLASSES = [
    "conflict", "danger", "loss_absence", "pressure", "safety",
    "teasing", "vanity",
]

# EMOTIC's 26 raw categories -> final emotion class.
# Engagement and Surprise have no slot in the final 14-class table and
# are not named in any merge rule -> dropped (excluded_pool).
# This mapping matches the user-provided "최종 감정 풀 - 14종" table
# (통합 원본 라벨 column): Esteem/Sympathy -> affection (not calm),
# Sensitivity -> aversion (not confusion).
EMOTIC_CATEGORY_TO_EMOTION = {
    "Happiness": "happiness",
    "Excitement": "happiness",
    "Pleasure": "happiness",
    "Anger": "anger",
    "Annoyance": "anger",
    "Disapproval": "anger",
    "Sadness": "sadness",
    "Suffering": "suffering",
    "Pain": "suffering",
    "Fatigue": "suffering",
    "Aversion": "aversion",
    "Disconnection": "disconnection",
    "Fear": "fear",
    "Disquietment": "fear",
    "Doubt/Confusion": "confusion",
    "Sensitivity": "aversion",
    "Embarrassment": "embarrassment",
    "Confidence": "confidence",
    "Anticipation": "anticipation",
    "Yearning": "yearning",
    "Affection": "affection",
    "Esteem": "affection",
    "Sympathy": "affection",
    "Peace": "calm",
    # Dropped - no documented slot in the final 14-class table.
    "Engagement": None,
    "Surprise": None,
}

# manual_images/<folder> -> (emotion_class_or_None, situation_class_or_None)
# Reconstructed by matching folder file counts against the Day 5
# "manual 감정(99건)" / "manual 상황(128건)" distribution tables - every
# folder below reproduces its documented count exactly except where noted.
MANUAL_FOLDER_TO_LABELS = {
    # situation-only folders (emotion field intentionally empty)
    "safety": (None, "safety"),
    "danger": (None, "danger"),
    "concert": (None, "conflict"),  # folder name predates the conflict relabel
    "teasing": (None, "teasing"),
    "pressure": (None, "pressure"),
    "superiority": (None, "pressure"),  # explicit merge rule in Day 5
    "vanity": (None, "vanity"),
    # dual-axis: "missing" carries both a situation and an emotion
    "missing": ("yearning", "loss_absence"),
    # emotion-only folders
    "elation": ("happiness", None),
    "euphoria": ("happiness", None),
    "manic": ("happiness", None),
    "joy": ("happiness", None),
    "aggression": ("anger", None),
    "jealousy": ("anger", None),
    "bullying": ("anger", None),
    "protest": ("anger", None),
    "nervous": ("fear", None),
    "exhausted": ("disconnection", None),
    "emptiness": ("disconnection", None),
    "alienation": ("disconnection", None),
    "bittersweet": ("yearning", None),
    "warmth": ("calm", None),
    "relief": ("calm", None),
    "hope": ("calm", None),
    "forgiveness": ("calm", None),
    # empathy has no slot in the fixed 14-class emotion scheme (2026-06-22) ->
    # dropped, same treatment as EMOTIC's Engagement/Surprise.
    "empathy": (None, None),
    "embarrassment": ("confusion", None),  # folder name, not the EMOTIC category
    # confirmed via user-provided final table: sadness's 통합원본라벨 includes
    # "despair" explicitly, so this is no longer unresolved.
    "despair": ("sadness", None),
}

# Folders kept for provenance but with no destination in the fixed schema -
# build_manual_labels.py routes these to manual_images_unresolved.csv.
NO_SCHEMA_SLOT_FOLDERS = {
    folder for folder, (e, s) in MANUAL_FOLDER_TO_LABELS.items() if e is None and s is None
}
