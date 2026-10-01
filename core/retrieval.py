"""Ứng dụng 3 — Tìm kiếm ảnh (CLIP + FAISS): tìm bằng câu mô tả hoặc bằng ảnh mẫu."""
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import ART_DIR, CLIP_MODEL, DEVICE, resolve_path


def _features(out) -> torch.Tensor:
    # transformers 4.x trả tensor; 5.x trả object có pooler_output đã chiếu sang không gian CLIP
    return out if torch.is_tensor(out) else out.pooler_output


class ClipEncoder:
    def __init__(self, model_name: str = CLIP_MODEL):
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        chunks = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(DEVICE)
            chunks.append(F.normalize(_features(self.model.get_image_features(**inputs)), dim=-1).cpu())
        return torch.cat(chunks).numpy().astype("float32")

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        feats = _features(self.model.get_text_features(**inputs))
        return F.normalize(feats, dim=-1).cpu().numpy().astype("float32")


def build_index(encoder: ClipEncoder, items: list[dict], out_dir: Path = ART_DIR / "retrieval") -> faiss.Index:
    """items: [{"path": <tương đối so với ROOT>, "label": ..., "source": ...}] → lưu index.faiss + meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = encoder.encode_images([Image.open(resolve_path(it["path"])) for it in items])
    index = faiss.IndexFlatIP(embs.shape[1])  # vector đã chuẩn hoá → inner product = cosine
    index.add(embs)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "meta.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return index


class ImageSearch:
    def __init__(self, index_dir: Path = ART_DIR / "retrieval", encoder: ClipEncoder | None = None):
        index_dir = Path(index_dir)
        self.encoder = encoder or ClipEncoder()
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.meta: list[dict] = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

    def _search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        scores, ids = self.index.search(query_vec, k)
        return [
            {"id": int(i), "score": round(float(s), 4), **self.meta[i]}
            for s, i in zip(scores[0], ids[0]) if i != -1
        ]

    def search_text(self, query: str, k: int = 8) -> list[dict]:
        return self._search(self.encoder.encode_texts([query]), k)

    def search_image(self, image: Image.Image, k: int = 8) -> list[dict]:
        return self._search(self.encoder.encode_images([image]), k)