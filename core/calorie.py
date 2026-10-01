"""AI 5 — Ước tính calo bằng CLIP zero-shot + cổng food/non-food + xác nhận người dùng."""
from __future__ import annotations

import json
import threading

import numpy as np

from config import DATA_DIR

NEGATIVE_PROMPTS = [
    "a non-food photo of a live animal",
    "a non-food photo of a person",
    "a non-food photo of a vehicle",
    "a non-food photo of a computer or electronic device",
    "a non-food photo of a landscape",
    "a non-food photo of a household object",
]


class FoodCalorieEstimator:
    def __init__(self):
        payload = json.loads((DATA_DIR / "nutrition_calorie.json").read_text(encoding="utf-8"))
        self.foods = payload["foods"]
        self.warning = payload["meta"]["warning"]
        self.keys = list(self.foods)
        self.encoder = None
        self.food_embeddings = None
        self.negative_embeddings = None
        self._lock = threading.Lock()

    def set_encoder(self, encoder):
        self.encoder = encoder
        self.food_embeddings = None
        self.negative_embeddings = None

    def _ensure_encoder(self):
        if self.encoder is None:
            from core.retrieval import ClipEncoder
            self.encoder = ClipEncoder()
        if self.food_embeddings is not None:
            return
        with self._lock:
            if self.food_embeddings is not None:
                return
            # Mỗi món có 3 prompt; lấy trung bình vector để giảm nhạy với một cách diễn đạt.
            food_vecs = []
            for key in self.keys:
                concept = self.foods[key]["concept"]
                prompts = [
                    f"a food photo of {concept}",
                    f"a close-up photo of {concept} served as food",
                    f"a plated dish containing {concept}",
                ]
                vec = self.encoder.encode_texts(prompts).mean(axis=0)
                vec /= np.linalg.norm(vec) + 1e-8
                food_vecs.append(vec)
            self.food_embeddings = np.stack(food_vecs).astype("float32")
            self.negative_embeddings = self.encoder.encode_texts(NEGATIVE_PROMPTS)

    @staticmethod
    def _softmax(x: np.ndarray, scale: float = 10.0) -> np.ndarray:
        z = x.astype("float64") * scale
        z -= z.max()
        e = np.exp(z)
        return (e / e.sum()).astype("float32")

    def food_options(self) -> list[dict]:
        return [
            {"label": k, "name": self.foods[k]["name_vi"], "kcal_per_100g": float(self.foods[k]["kcal_per_100g"])}
            for k in self.keys
        ]

    def recalculate(self, food_label: str, grams: float) -> dict:
        if food_label not in self.foods:
            raise ValueError("Món ăn không có trong bảng tham khảo.")
        grams = float(max(1.0, min(float(grams), 3000.0)))
        info = self.foods[food_label]
        kcal100 = float(info["kcal_per_100g"])
        kcal = kcal100 * grams / 100.0
        return {
            "food": {"label": food_label, "name": info["name_vi"]},
            "portion_grams": round(grams, 1),
            "kcal_per_100g": kcal100,
            "estimated_kcal": round(kcal),
            "estimated_range_kcal": [round(kcal * 0.75), round(kcal * 1.25)],
            "warning": self.warning,
        }

    def estimate(self, image, grams: float = 200.0, top_k: int = 5) -> dict:
        self._ensure_encoder()
        grams = float(max(1.0, min(float(grams), 3000.0)))
        top_k = int(max(1, min(int(top_k), 5)))

        q = self.encoder.encode_images([image])[0]
        food_sim = self.food_embeddings @ q
        neg_sim = self.negative_embeddings @ q
        probs = self._softmax(food_sim, 10.0)
        order = np.argsort(-food_sim)[:top_k]

        predictions = []
        for i in order:
            key = self.keys[int(i)]
            info = self.foods[key]
            predictions.append({
                "label": key,
                "name": info["name_vi"],
                "match_score": round(float(probs[int(i)]), 4),
                "similarity": round(float(food_sim[int(i)]), 4),
                "kcal_per_100g": float(info["kcal_per_100g"]),
            })

        best = predictions[0]
        second_similarity = predictions[1]["similarity"] if len(predictions) > 1 else -1.0
        best_neg = float(np.max(neg_sim))
        is_food = best["similarity"] >= 0.17 and best["similarity"] >= best_neg - 0.005
        margin = best["similarity"] - second_similarity
        uncertain = (not is_food) or best["similarity"] < 0.20 or margin < 0.012

        if is_food:
            calc = self.recalculate(best["label"], grams)
            estimated_kcal = calc["estimated_kcal"]
            estimated_range = calc["estimated_range_kcal"]
        else:
            estimated_kcal = None
            estimated_range = None

        return {
            "is_food": bool(is_food),
            "food": {
                "label": best["label"],
                "name": best["name"],
                "match_score": best["match_score"],
                "similarity": best["similarity"],
            },
            "uncertain": bool(uncertain),
            "portion_grams": round(grams, 1),
            "kcal_per_100g": best["kcal_per_100g"],
            "estimated_kcal": estimated_kcal,
            "estimated_range_kcal": estimated_range,
            "top_predictions": predictions,
            "correction_options": self.food_options(),
            "food_vs_nonfood_margin": round(best["similarity"] - best_neg, 4),
            "method": "CLIP zero-shot ensemble",
            "warning": self.warning,
        }
