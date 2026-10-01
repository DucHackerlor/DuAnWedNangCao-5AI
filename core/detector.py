"""Ứng dụng 2 — Phát hiện đối tượng YOLO11 với bước kiểm tra lại ca khó."""
from __future__ import annotations

from collections import Counter
from statistics import mean

from PIL import Image
from ultralytics import YOLO

from config import DEVICE, YOLO_IMGSZ, YOLO_RETRY_BELOW, YOLO_RETRY_IMGSZ, YOLO_WEIGHTS
from core.labels import coco_vi, confidence_band


class ObjectDetector:
    def __init__(self, weights: str = YOLO_WEIGHTS):
        self.model = YOLO(weights)
        self.device = 0 if DEVICE == "cuda" else "cpu"
        self.weights = weights

    def _predict(self, image: Image.Image, conf: float, iou: float, imgsz: int, augment: bool):
        return self.model.predict(
            image.convert("RGB"),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=100,
            device=self.device,
            verbose=False,
            augment=augment,
            agnostic_nms=False,
        )[0]

    @staticmethod
    def _quality(result) -> float:
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0
        scores = [float(v) for v in result.boxes.conf.tolist()]
        high = sum(s >= 0.50 for s in scores)
        return high * 2.0 + mean(scores) + min(len(scores), 10) * 0.05

    def detect(self, image: Image.Image, conf: float = 0.30, iou: float = 0.50, max_det: int = 100):
        del max_det  # giữ tương thích chữ ký cũ; model đã giới hạn 100 ở _predict.
        image = image.convert("RGB")
        first = self._predict(image, conf=conf, iou=iou, imgsz=YOLO_IMGSZ, augment=False)

        scores_first = [] if first.boxes is None else [float(v) for v in first.boxes.conf.tolist()]
        uncertain = (not scores_first) or any(s < YOLO_RETRY_BELOW for s in scores_first)
        used_retry = False
        result = first

        # Chỉ ca khó mới chạy lượt 2 ở độ phân giải cao + TTA. Nhờ vậy đa số ảnh vẫn nhanh.
        if uncertain:
            retry_conf = max(0.20, min(conf, 0.30))
            second = self._predict(image, conf=retry_conf, iou=0.50, imgsz=YOLO_RETRY_IMGSZ, augment=True)
            if self._quality(second) > self._quality(first):
                result = second
                used_retry = True

        boxes = result.boxes
        detections = []
        if boxes is not None:
            for b, s, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                label_en = result.names[int(c)]
                score = float(s)
                detections.append({
                    "label": label_en,                    # tương thích code cũ
                    "label_en": label_en,
                    "label_vi": coco_vi(label_en),
                    "score": round(score, 4),
                    "confidence_level": confidence_band(score),
                    "needs_review": score < 0.50,
                    "box_xyxy": [round(v, 1) for v in b],
                })

        annotated = Image.fromarray(result.plot()[..., ::-1])
        summary = dict(Counter(d["label_vi"] for d in detections))
        low_count = sum(d["needs_review"] for d in detections)
        warnings = []
        if not detections:
            warnings.append("Không phát hiện được đối tượng đủ tin cậy trong ảnh.")
        if low_count:
            warnings.append(f"Có {low_count} đối tượng độ tin cậy thấp; nên kiểm tra lại bằng mắt.")

        return {
            "detections": detections,
            "summary": summary,
            "warnings": warnings,
            "used_retry": used_retry,
            "model": str(self.weights),
        }, annotated
