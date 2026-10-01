"""API regression tests. Không tải model thật; dùng fake models để test contract, validation và lỗi."""
import io
import os

os.environ["ENABLED_MODELS"] = ""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main


class FakeClassifier:
    def predict(self, image, top_k=3):
        return {
            "predictions": [{"label": "roses", "label_vi": "hoa hồng", "score": 0.91}][:top_k],
            "confident": True,
            "reason": None,
            "agreement": True,
            "method": "fake",
            "margin": 0.5,
        }


class FakeDetector:
    def detect(self, image, conf=0.30):
        return {
            "detections": [{
                "label": "dog", "label_en": "dog", "label_vi": "chó", "score": 0.82,
                "confidence_level": "cao", "needs_review": False, "box_xyxy": [0, 0, 10, 10]
            }],
            "summary": {"chó": 1}, "warnings": [], "used_retry": False, "model": "fake"
        }, image


class FakeRetrieval:
    meta = [{"path": "x.jpg", "label": "dog"}]
    def normalize_query(self, q): return "dog" if q == "chó" else q
    def search_text(self, query, k=8):
        return [{"id": 0, "score": 0.31, "quality": "cao", "label": "dog", "path": "x.jpg"}]
    def search_image(self, image, k=8):
        return [{"id": 0, "score": 0.31, "quality": "cao", "label": "dog", "path": "x.jpg"}]


class FakeBot:
    def stream(self, message, history=None):
        return [{"source": "doi_tra.md", "text": "7 ngày", "score": 0.9}], iter(["Được ", "7 ngày."])
    def answer(self, message, history=None):
        return {"answer": "Được 7 ngày.\nNguồn: [doi_tra.md]", "sources": []}


class FakeCalorie:
    def food_options(self):
        return [
            {"label": "crab_dish", "name": "Cua / món cua", "kcal_per_100g": 170.0},
            {"label": "peking_duck", "name": "Vịt quay", "kcal_per_100g": 337.0},
        ]
    def recalculate(self, food_label, grams):
        if food_label != "crab_dish":
            raise ValueError("Món ăn không có trong bảng tham khảo.")
        kcal = 170 * float(grams) / 100
        return {
            "food": {"label": food_label, "name": "Cua / món cua"},
            "portion_grams": float(grams), "kcal_per_100g": 170.0,
            "estimated_kcal": round(kcal), "estimated_range_kcal": [round(kcal*.75), round(kcal*1.25)],
            "warning": "estimate",
        }
    def estimate(self, image, grams=200, top_k=5):
        return {
            "is_food": True,
            "food": {"label": "crab_dish", "name": "Cua / món cua", "match_score": 0.74, "similarity": 0.29},
            "uncertain": False,
            "portion_grams": float(grams), "kcal_per_100g": 170.0,
            "estimated_kcal": round(170 * float(grams) / 100),
            "estimated_range_kcal": [255, 425],
            "top_predictions": [{"label": "crab_dish", "name": "Cua / món cua", "match_score": 0.74, "similarity": 0.29, "kcal_per_100g": 170.0}],
            "correction_options": self.food_options(),
            "warning": "estimate",
        }


def png_bytes(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        main.MODELS.update(
            classifier=FakeClassifier(), detector=FakeDetector(), retrieval=FakeRetrieval(),
            llm=FakeBot(), calorie=FakeCalorie()
        )
        yield c
        main.MODELS.clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["version"] == "2.0.0"


def test_classify_ok(client):
    r = client.post("/api/classify", files={"file": ("a.png", png_bytes(), "image/png")}, data={"top_k": 1})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["label_vi"] == "hoa hồng"


def test_rejects_non_image(client):
    r = client.post("/api/classify", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_tiny_image(client):
    r = client.post("/api/classify", files={"file": ("tiny.png", png_bytes((8, 8)), "image/png")})
    assert r.status_code == 400


def test_detect_returns_vietnamese_label_and_image(client):
    r = client.post("/api/detect", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert data["image"].startswith("data:image/jpeg;base64,")
    assert data["detections"][0]["label_vi"] == "chó"


def test_search_translates_common_vietnamese_query(client):
    r = client.post("/api/search/text", json={"query": "chó", "k": 5})
    assert r.status_code == 200
    assert r.json()["query_used"] == "dog"
    assert r.json()["results"][0]["label"] == "dog"


def test_empty_query_returns_422(client):
    assert client.post("/api/search/text", json={"query": ""}).status_code == 422


def test_chat_stream_events(client):
    with client.stream("POST", "/api/chat", json={"message": "Đổi trả?"}) as r:
        body = "".join(r.iter_text())
    assert '"type": "sources"' in body and '"type": "done"' in body and "7 ngày" in body


def test_calorie_ok(client):
    r = client.post("/api/calorie", files={"file": ("food.png", png_bytes(), "image/png")}, data={"grams": 200})
    assert r.status_code == 200
    assert r.json()["food"]["label"] == "crab_dish"
    assert r.json()["estimated_kcal"] == 340


def test_calorie_recalculate(client):
    r = client.post("/api/calorie/recalculate", json={"food_label": "crab_dish", "grams": 250})
    assert r.status_code == 200
    assert r.json()["estimated_kcal"] == 425


def test_calorie_invalid_grams(client):
    r = client.post("/api/calorie/recalculate", json={"food_label": "crab_dish", "grams": 0})
    assert r.status_code == 422
