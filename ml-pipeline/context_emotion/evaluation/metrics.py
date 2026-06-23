from typing import List

from sklearn.metrics import classification_report, confusion_matrix

from context_emotion.common.constants import EMOTION_CLASSES


def print_eval_report(y_true: List[int], y_pred: List[int]) -> None:
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(EMOTION_CLASSES))),
        target_names=EMOTION_CLASSES,
        zero_division=0,
    ))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_CLASSES)))))
