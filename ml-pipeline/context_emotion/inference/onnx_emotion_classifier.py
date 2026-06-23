"""Minimal ONNX inference wrapper, mirroring flashlight's
inference/onnx_mouse_detector.py shape."""
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from context_emotion.common.constants import EMOTION_CLASSES
from context_emotion.training.train_loop import IMAGENET_MEAN, IMAGENET_STD

INPUT_SIZE = (224, 224)


class OnnxEmotionClassifier:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(INPUT_SIZE)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def predict(self, image: Image.Image) -> Tuple[str, np.ndarray]:
        logits = self.session.run(None, {self.input_name: self._preprocess(image)})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return EMOTION_CLASSES[int(probs.argmax())], probs
