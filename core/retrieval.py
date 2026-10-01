"""Ứng dụng 3 — Tìm kiếm ảnh CLIP + FAISS, hỗ trợ truy vấn Việt phổ biến và lọc kết quả yếu."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import ART_DIR, CLIP_MODEL, DEVICE, SEARCH_MIN_SCORE, resolve_path
from core.labels import normalize_search_query, token_overlap


def _features(out) -> torch.Tensor:
    return out if torch.is_tensor(out) else out.pooler_output


class ClipEncoder:
    def __init__(self, model_name: str = CLIP_MODEL):
        self.model_name = model_name
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for it in items:
        with Image.open(resolve_path(it["path"])) as im:
            images.append(im.convert("RGB").copy())
    embs = encoder.encode_images(images)
    index = faiss.IndexFlatIP(embs.shape[1])
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
        if self.index.ntotal != len(self.meta):
            raise RuntimeError(f"FAISS index có {self.index.ntotal} vector nhưng meta có {len(self.meta)} mục.")

    @staticmethod
    def normalize_query(query: str) -> str:
        return normalize_search_query(query)

    def _raw_search(self, query_vec: np.ndarray, k: int) -> list[tuple[float, int]]:
        k = max(1, min(int(k), int(self.index.ntotal)))
        scores, ids = self.index.search(query_vec.astype("float32"), k)
        return [(float(s), int(i)) for s, i in zip(scores[0], ids[0]) if i != -1]

    @staticmethod
    def _quality(score: float) -> str:
        if score >= 0.28:
            return "cao"
        if score >= 0.20:
            return "trung bình"
        return "thấp"

    def search_text(self, query: str, k: int = 8) -> list[dict]:
        query_used = self.normalize_query(query)
        q = self.encoder.encode_texts([query_used])
        # Lấy nhiều ứng viên rồi rerank nhẹ theo nhãn YOLO/Flowers đi kèm metadata.
        candidates = self._raw_search(q, min(max(k * 4, 20), self.index.ntotal))
        reranked = []
        for score, idx in candidates:
            meta = self.meta[idx]
            label = str(meta.get("label", ""))
            bonus = min(token_overlap(query_used, label) * 0.035, 0.07)
            reranked.append((score + bonus, score, idx))
        reranked.sort(reverse=True)

        out = []
        for adjusted, raw, idx in reranked[:k]:
            if raw < SEARCH_MIN_SCORE and out:
                continue
            out.append({
                "id": idx,
                "score": round(raw, 4),
                "adjusted_score": round(adjusted, 4),
                "quality": self._quality(raw),
                "query_used": query_used,
                **self.meta[idx],
            })
        return out

    def search_image(self, image: Image.Image, k: int = 8) -> list[dict]:
        rows = self._raw_search(self.encoder.encode_images([image]), k)
        return [{
            "id": idx,
            "score": round(score, 4),
            "quality": self._quality(score),
            **self.meta[idx],
        } for score, idx in rows]
