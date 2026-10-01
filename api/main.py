"""FastAPI backend: 5 AI, có kiểm tra độ chắc chắn và xử lý model nặng trong threadpool."""
from __future__ import annotations

import base64
import importlib
import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from config import CORS_ORIGINS, DEVICE, ENABLED_MODELS, MAX_UPLOAD_MB, ROOT, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
MODELS: dict = {}
MODEL_ERRORS: dict[str, str] = {}
MODEL_LOAD_SECONDS: dict[str, float] = {}

LOADERS = {
    "classifier": ("core.classifier", "ImageClassifier"),
    "detector": ("core.detector", "ObjectDetector"),
    "retrieval": ("core.retrieval", "ImageSearch"),
    "llm": ("core.llm", "RAGChatbot"),
    "calorie": ("core.calorie", "FoodCalorieEstimator"),
}


def _load_models():
    MODELS.clear()
    MODEL_ERRORS.clear()
    MODEL_LOAD_SECONDS.clear()

    for name, (module, cls) in LOADERS.items():
        if name not in ENABLED_MODELS:
            continue
        t0 = time.perf_counter()
        try:
            MODELS[name] = getattr(importlib.import_module(module), cls)()
            MODEL_LOAD_SECONDS[name] = round(time.perf_counter() - t0, 2)
            log.info("loaded %s in %.1fs", name, MODEL_LOAD_SECONDS[name])
        except Exception as exc:
            MODEL_ERRORS[name] = f"{type(exc).__name__}: {exc}"
            MODEL_LOAD_SECONDS[name] = round(time.perf_counter() - t0, 2)
            log.exception("cannot load %s: %s", name, exc)

    # Một CLIP dùng chung cho search + classifier verification + calorie để đỡ RAM.
    if "retrieval" in MODELS:
        encoder = MODELS["retrieval"].encoder
        if "classifier" in MODELS:
            try:
                MODELS["classifier"].set_clip_encoder(encoder)
                log.info("classifier: reused retrieval CLIP encoder")
            except Exception as exc:
                log.warning("classifier could not reuse CLIP: %s", exc)
        if "calorie" in MODELS:
            try:
                MODELS["calorie"].set_encoder(encoder)
                log.info("calorie: reused retrieval CLIP encoder")
            except Exception as exc:
                log.warning("calorie could not reuse CLIP: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    MODELS.clear()


app = FastAPI(title="AI Web Apps API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _require(name: str):
    if name not in MODELS:
        detail = MODEL_ERRORS.get(name)
        if detail:
            raise HTTPException(503, f"Mô hình '{name}' chưa sẵn sàng: {detail}")
        raise HTTPException(503, f"Mô hình '{name}' chưa được nạp (xem /api/health)")
    return MODELS[name]


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "File ảnh rỗng.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Ảnh vượt quá {MAX_UPLOAD_MB} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        if image.width < 16 or image.height < 16:
            raise HTTPException(400, "Ảnh quá nhỏ để AI xử lý.")
        return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File không phải ảnh hợp lệ (jpg, png, webp)")


def _to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health():
    names = sorted(set(LOADERS) | set(ENABLED_MODELS))
    return {
        "status": "ok",
        "version": app.version,
        "device": DEVICE,
        "models": {m: m in MODELS for m in names},
        "enabled": sorted(ENABLED_MODELS),
        "load_seconds": MODEL_LOAD_SECONDS,
        "errors": MODEL_ERRORS,
    }


# ---------- 1. Phân loại hoa ----------
@app.post("/api/classify")
async def classify(file: UploadFile = File(...), top_k: int = Form(3)):
    model = _require("classifier")
    image = await _read_image(file)
    t0 = time.perf_counter()
    result = await run_in_threadpool(model.predict, image, max(1, min(top_k, 5)))
    return {**result, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------- 2. Phát hiện đối tượng ----------
@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.30)):
    model = _require("detector")
    image = await _read_image(file)
    t0 = time.perf_counter()
    result, annotated = await run_in_threadpool(model.detect, image, min(max(conf, 0.05), 0.95))
    return {**result, "image": _to_base64(annotated), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------- 3. Tìm kiếm ảnh ----------
class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    k: int = Field(8, ge=1, le=24)


def _with_urls(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "path"} | {"url": f"/api/gallery/{r['id']}"} for r in results]


@app.post("/api/search/text")
async def search_text(q: TextQuery):
    engine = _require("retrieval")
    results = await run_in_threadpool(engine.search_text, q.query, q.k)
    return {
        "query": q.query,
        "query_used": engine.normalize_query(q.query),
        "results": _with_urls(results),
        "warning": None if results else "Không có kết quả đủ liên quan; hãy mô tả cụ thể hơn.",
    }


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(8)):
    engine = _require("retrieval")
    image = await _read_image(file)
    results = await run_in_threadpool(engine.search_image, image, max(1, min(k, 24)))
    return {"results": _with_urls(results)}


@app.get("/api/gallery/{item_id}")
def gallery(item_id: int):
    engine = _require("retrieval")
    if not 0 <= item_id < len(engine.meta):
        raise HTTPException(404, "Không có ảnh này")
    return FileResponse(resolve_path(engine.meta[item_id]["path"]), headers={"Cache-Control": "public, max-age=3600"})


# ---------- 4. Chatbot RAG ----------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list)


@app.post("/api/chat")
def chat(req: ChatRequest):
    bot = _require("llm")
    contexts, tokens = bot.stream(req.message, req.history)

    def events():
        yield f"data: {json.dumps({'type': 'sources', 'items': contexts}, ensure_ascii=False)}\n\n"
        for piece in tokens:
            yield f"data: {json.dumps({'type': 'token', 'text': piece}, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        events(), media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/sync")
def chat_sync(req: ChatRequest):
    return _require("llm").answer(req.message, req.history)


# ---------- 5. Ước tính calo ----------
@app.post("/api/calorie")
async def calorie_estimate(file: UploadFile = File(...), grams: float = Form(200), top_k: int = Form(5)):
    if not 1 <= grams <= 3000:
        raise HTTPException(400, "Khối lượng phải nằm trong khoảng 1-3000 gram.")
    estimator = _require("calorie")
    image = await _read_image(file)
    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(estimator.estimate, image, grams, max(1, min(top_k, 5)))
    except Exception as exc:
        MODEL_ERRORS["calorie"] = f"{type(exc).__name__}: {exc}"
        log.exception("calorie inference failed: %s", exc)
        raise HTTPException(503, f"AI calo chưa xử lý được ảnh: {type(exc).__name__}: {exc}")
    return {**result, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


class RecalculateRequest(BaseModel):
    food_label: str = Field(..., min_length=1, max_length=80)
    grams: float = Field(..., ge=1, le=3000)


@app.post("/api/calorie/recalculate")
def calorie_recalculate(req: RecalculateRequest):
    try:
        return _require("calorie").recalculate(req.food_label, req.grams)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/calorie/foods")
def calorie_foods():
    return {"foods": _require("calorie").food_options()}


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
