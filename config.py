"""Cấu hình tập trung. Mọi giá trị đều ghi đè được bằng biến môi trường."""
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO11s chính xác hơn YOLO11n nhưng vẫn đủ nhẹ để chạy demo trên CPU.
# Muốn quay lại bản nhẹ: đặt YOLO_WEIGHTS=artifacts/detector/yolo11n.pt
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11s.pt")
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
YOLO_RETRY_IMGSZ = int(os.environ.get("YOLO_RETRY_IMGSZ", "832"))
YOLO_RETRY_BELOW = float(os.environ.get("YOLO_RETRY_BELOW", "0.60"))

CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct",
)

CLASSIFIER_MIN_CONFIDENCE = float(os.environ.get("CLASSIFIER_MIN_CONFIDENCE", "0.52"))
CLASSIFIER_MIN_MARGIN = float(os.environ.get("CLASSIFIER_MIN_MARGIN", "0.08"))
SEARCH_MIN_SCORE = float(os.environ.get("SEARCH_MIN_SCORE", "0.16"))
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.32"))

# Bật/tắt từng mô hình để tiết kiệm bộ nhớ.
ENABLED_MODELS = {
    m.strip()
    for m in os.environ.get("ENABLED_MODELS", "classifier,detector,retrieval,llm,calorie").split(",")
    if m.strip()
}

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
