"""Ứng dụng 1 — Phân loại 5 loài hoa: ResNet-18 + CLIP kiểm chứng khi có sẵn."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from config import ART_DIR, CLASSIFIER_MIN_CONFIDENCE, CLASSIFIER_MIN_MARGIN, DEVICE
from core.labels import flower_vi

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.18, 0.18, 0.18),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

FLOWER_PROMPTS = {
    "daisy": "a close-up photo of a daisy flower",
    "dandelion": "a close-up photo of a dandelion flower",
    "roses": "a close-up photo of a rose flower",
    "rose": "a close-up photo of a rose flower",
    "sunflowers": "a close-up photo of a sunflower",
    "sunflower": "a close-up photo of a sunflower",
    "tulips": "a close-up photo of a tulip flower",
    "tulip": "a close-up photo of a tulip flower",
}
NEGATIVE_PROMPTS = [
    "a photo of a person",
    "a photo of a dog or cat",
    "a photo of food on a plate",
    "a photo of a vehicle",
    "a photo of an everyday object that is not a flower",
]


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def _softmax_np(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = x.astype("float64") * scale
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum()).astype("float32")


class ImageClassifier:
    def __init__(self, model_dir: Path = ART_DIR / "classifier",
                 min_confidence: float = CLASSIFIER_MIN_CONFIDENCE,
                 min_margin: float = CLASSIFIER_MIN_MARGIN):
        model_dir = Path(model_dir)
        self.classes: list[str] = json.loads((model_dir / "classes.json").read_text(encoding="utf-8"))
        self.model = build_model(len(self.classes), pretrained=False)
        state = torch.load(model_dir / "model.pt", map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
        self.min_confidence = min_confidence
        self.min_margin = min_margin

        self.clip_encoder = None
        self._clip_flower_embs = None
        self._clip_negative_embs = None

    def set_clip_encoder(self, encoder):
        """Dùng chung CLIP với AI tìm kiếm ảnh, không nạp model thứ hai."""
        self.clip_encoder = encoder
        prompts = [FLOWER_PROMPTS.get(c, f"a close-up photo of a {c} flower") for c in self.classes]
        self._clip_flower_embs = encoder.encode_texts(prompts)
        self._clip_negative_embs = encoder.encode_texts(NEGATIVE_PROMPTS)

    @torch.inference_mode()
    def _resnet_probs(self, image: Image.Image) -> np.ndarray:
        x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        return self.model(x).softmax(dim=-1)[0].cpu().numpy().astype("float32")

    def _clip_scores(self, image: Image.Image):
        if self.clip_encoder is None or self._clip_flower_embs is None:
            return None, None, None
        q = self.clip_encoder.encode_images([image])[0]
        flower_sim = self._clip_flower_embs @ q
        negative_sim = self._clip_negative_embs @ q
        flower_probs = _softmax_np(flower_sim, scale=12.0)
        return flower_probs, flower_sim, negative_sim

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        image = image.convert("RGB")
        resnet = self._resnet_probs(image)
        clip_probs, flower_sim, negative_sim = self._clip_scores(image)

        method = "ResNet-18"
        agreement = None
        if clip_probs is not None:
            r_top = int(np.argmax(resnet))
            c_top = int(np.argmax(clip_probs))
            agreement = r_top == c_top
            # Khi 2 model không đồng ý, ưu tiên CLIP hơn để giảm lỗi do ResNet train ít epoch.
            clip_weight = 0.60 if (not agreement or float(resnet[r_top]) < 0.70) else 0.40
            probs = (1.0 - clip_weight) * resnet + clip_weight * clip_probs
            method = "ResNet-18 + CLIP verification"
        else:
            probs = resnet

        order = np.argsort(-probs)[:min(top_k, len(self.classes))]
        predictions = [{
            "label": self.classes[int(i)],
            "label_vi": flower_vi(self.classes[int(i)]),
            "score": round(float(probs[int(i)]), 4),
            "resnet_score": round(float(resnet[int(i)]), 4),
            **({"clip_score": round(float(clip_probs[int(i)]), 4)} if clip_probs is not None else {}),
        } for i in order]

        best = predictions[0]["score"]
        second = predictions[1]["score"] if len(predictions) > 1 else 0.0
        margin = best - second
        confident = best >= self.min_confidence and margin >= self.min_margin
        reason = None

        if clip_probs is not None:
            best_flower_sim = float(np.max(flower_sim))
            best_negative_sim = float(np.max(negative_sim))
            # Nếu ảnh giống "không phải hoa" hơn các prompt hoa, từ chối gán nhãn cưỡng ép.
            if best_negative_sim > best_flower_sim + 0.01:
                confident = False
                reason = "Ảnh có vẻ không thuộc 5 loài hoa đã học."
            elif agreement is False and margin < 0.15:
                confident = False
                reason = "ResNet và CLIP chưa đồng thuận đủ mạnh."

        if not confident and reason is None:
            reason = "Độ chắc chắn hoặc khoảng cách giữa hai dự đoán đầu chưa đủ lớn."

        return {
            "predictions": predictions,
            "confident": bool(confident),
            "reason": reason,
            "agreement": agreement,
            "method": method,
            "margin": round(float(margin), 4),
        }
