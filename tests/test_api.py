"""Test API không cần GPU hay tải mô hình: thay mô hình thật bằng bản giả (dependency injection).
Chạy: ENABLED_MODELS= pytest -q"""
import io
import os

os.environ["ENABLED_MODELS"] = ""  # không nạp mô hình thật khi khởi động

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main


class FakeClassifier:
    def predict(self, image, top_k=3):
        return {"predictions": [{"label": "roses", "score": 0.9}][:top_k], "confident": True}


class FakeDetector:
    def detect(self, image, conf=0.25):
        return {"detections": [{"label": "person", "score": 0.8, "box_xyxy": [0, 0, 10, 10]}],
                "summary": {"person": 1}}, image


class FakeBot:
    def stream(self, message, history=None):
        return [{"source": "doi_tra.md", "text": "7 ngày", "score": 0.9}], iter(["Được ", "7 ngày."])

    def answer(self, message, history=None):
        return {"answer": "Được 7 ngày.", "sources": []}


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        main.MODELS.update(classifier=FakeClassifier(), detector=FakeDetector(), llm=FakeBot())
        yield c
        main.MODELS.clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_classify_ok(client):
    r = client.post("/api/classify", files={"file": ("a.png", png_bytes(), "image/png")}, data={"top_k": 1})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["label"] == "roses"


def test_classify_rejects_non_image(client):
    r = client.post("/api/classify", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_detect_returns_annotated_image(client):
    r = client.post("/api/detect", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["image"].startswith("data:image/jpeg;base64,")


def test_model_not_loaded_returns_503(client):
    r = client.post("/api/search/text", json={"query": "a dog"})
    assert r.status_code == 503


def test_empty_query_returns_422(client):
    assert client.post("/api/search/text", json={"query": ""}).status_code == 422


def test_chat_stream_events(client):
    with client.stream("POST", "/api/chat", json={"message": "Đổi trả?"}) as r:
        body = "".join(r.iter_text())
    assert '"type": "sources"' in body and '"type": "done"' in body and "7 ngày" in body