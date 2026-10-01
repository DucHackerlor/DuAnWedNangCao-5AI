"""Ước tính calo bằng CLIP zero-shot trên tập nhãn mở rộng.

Khắc phục hạn chế của Food-101 closed-set: nếu ảnh là cua nguyên con nhưng Food-101
không có lớp 'crab', softmax vẫn buộc model chọn nhầm một lớp khác với độ tin cậy cao.
Bản này dùng CLIP + các mô tả món ăn rộng hơn (có cua, tôm, cá, món Việt...) và có
cờ uncertain khi ảnh không khớp đủ rõ.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from config import DATA_DIR


class FoodCalorieEstimator:
    def __init__(self):
        payload = json.loads((DATA_DIR / "nutrition_calorie.json").read_text(encoding="utf-8"))
        self.foods = payload["foods"]
        self.warning = payload["meta"]["warning"]
        self.keys = list(self.foods)
        self.prompts = [self.foods[k]["prompt"] for k in self.keys]

        self.encoder = None
        self.text_embeddings = None
        self._lock = threading.Lock()

    def set_encoder(self, encoder):
        """Dùng lại CLIP encoder của AI tìm kiếm ảnh để không tốn thêm RAM/model."""
        self.encoder = encoder
        self.text_embeddings = None

    def _ensure_encoder(self):
        if self.encoder is None:
            from core.retrieval import ClipEncoder
            self.encoder = ClipEncoder()

        if self.text_embeddings is None:
            with self._lock:
                if self.text_embeddings is None:
                    self.text_embeddings = self.encoder.encode_texts(self.prompts)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def estimate(self, image, grams: float = 200.0, top_k: int = 3) -> dict:
        self._ensure_encoder()

        grams = float(max(1.0, min(float(grams), 3000.0)))
        top_k = int(max(1, min(int(top_k), 5)))

        image_vec = self.encoder.encode_images([image])[0]
        similarities = self.text_embeddings @ image_vec

        # Xác suất chỉ để xếp hạng trong danh sách nhãn hiện có; không gọi là
        # "độ tin cậy tuyệt đối". Scale vừa phải để tránh 99.9% giả tạo.
        probs = self._softmax(similarities * 10.0)
        order = np.argsort(-similarities)[:top_k]

        predictions = []
        for i in order:
            key = self.keys[int(i)]
            info = self.foods[key]
            predictions.append({
                "label": key,
                "name": info["name_vi"],
                "confidence": round(float(probs[int(i)]), 4),
                "similarity": round(float(similarities[int(i)]), 4),
                "kcal_per_100g": float(info["kcal_per_100g"]),
            })

        best = predictions[0]
        second_sim = predictions[1]["similarity"] if len(predictions) > 1 else -1.0
        margin = best["similarity"] - second_sim

        # CLIP cosine thường nằm khoảng 0.15-0.4 tùy ảnh/prompt.
        uncertain = best["similarity"] < 0.18 or margin < 0.012

        kcal = best["kcal_per_100g"] * grams / 100.0
        # Khoảng rộng hơn để phản ánh sai số công thức và khẩu phần.
        low, high = kcal * 0.75, kcal * 1.25

        return {
            "food": {
                "label": best["label"],
                "name": best["name"],
                "confidence": best["confidence"],
                "similarity": best["similarity"],
            },
            "uncertain": bool(uncertain),
            "portion_grams": round(grams, 1),
            "kcal_per_100g": best["kcal_per_100g"],
            "estimated_kcal": round(kcal),
            "estimated_range_kcal": [round(low), round(high)],
            "top_predictions": predictions,
            "method": "CLIP zero-shot",
            "warning": self.warning,
        }
