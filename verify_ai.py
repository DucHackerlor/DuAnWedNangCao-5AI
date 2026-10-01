"""Kiểm tra thực tế 5 AI trên dữ liệu local và lưu báo cáo JSON.

Chạy khi đã có dataset/artifacts:
    source ~/.venv/Scripts/activate
    python verify_ai.py

Script không sửa model. Nó chỉ đo nhanh các kiểm tra sanity/regression để trước khi nộp bài
biết chức năng nào đang yếu hoặc thiếu file.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from config import ART_DIR, DATA_DIR, ROOT

REPORT = {"checks": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def add(name, status, **data):
    REPORT["checks"][name] = {"status": status, **data}
    icon = "✅" if status == "pass" else "⚠️" if status == "warn" else "❌"
    print(f"{icon} {name}: {data}")


def check_files():
    required = [
        ART_DIR / "classifier" / "model.pt",
        ART_DIR / "classifier" / "classes.json",
        ART_DIR / "retrieval" / "index.faiss",
        ART_DIR / "retrieval" / "meta.json",
        DATA_DIR / "kb",
        DATA_DIR / "nutrition_calorie.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    add("files", "pass" if not missing else "fail", missing=missing)


def check_classifier(max_images=120):
    flower_dir = DATA_DIR / "flowers" / "flower_photos"
    split_file = ART_DIR / "classifier" / "split.json"
    if not flower_dir.exists() or not split_file.exists():
        add("classifier", "warn", reason="Thiếu Flowers dataset hoặc split.json; bỏ qua accuracy test.")
        return

    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    import torch
    import core.classifier as clf

    base = ImageFolder(flower_dir)
    test_idx = json.loads(split_file.read_text(encoding="utf-8"))["test"]
    test_idx = test_idx[:max_images]
    ds = Subset(ImageFolder(flower_dir, transform=clf.EVAL_TF), test_idx)

    model = clf.ImageClassifier()
    correct = 0
    total = 0
    for x, y in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
        with torch.inference_mode():
            probs = model.model(x.to(model.model.fc.weight.device)).softmax(-1)
        correct += (probs.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    acc = correct / max(total, 1)
    add("classifier", "pass" if acc >= 0.70 else "warn", accuracy=round(acc, 3), samples=total,
        note="Raw ResNet test; runtime còn có CLIP verification nên có cơ chế từ chối ca không chắc.")


def _coco_names():
    return [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
        'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
        'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
        'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]


def check_detector(max_images=20):
    img_dir = DATA_DIR / "coco128" / "images" / "train2017"
    label_dir = DATA_DIR / "coco128" / "labels" / "train2017"
    if not img_dir.exists() or not label_dir.exists():
        add("detector", "warn", reason="Thiếu COCO128; bỏ qua ground-truth sanity test.")
        return

    from core.detector import ObjectDetector
    names = _coco_names()
    images = sorted(img_dir.glob("*.jpg"))
    random.Random(42).shuffle(images)
    images = images[:max_images]
    det = ObjectDetector()
    gt_labels, hit = 0, 0
    for p in images:
        gt_file = label_dir / (p.stem + ".txt")
        expected = set()
        if gt_file.exists():
            for line in gt_file.read_text().splitlines():
                if line.strip():
                    expected.add(names[int(line.split()[0])])
        pred, _ = det.detect(Image.open(p), conf=0.25)
        got = {d["label_en"] for d in pred["detections"] if d["score"] >= 0.25}
        hit += len(expected & got)
        gt_labels += len(expected)
    recall = hit / max(gt_labels, 1)
    add("detector", "pass" if recall >= 0.55 else "warn", class_recall=round(recall, 3), images=len(images), gt_classes=gt_labels,
        note="Sanity metric trên COCO128; không phải mAP chuẩn.")


def check_retrieval():
    try:
        from core.retrieval import ImageSearch
        engine = ImageSearch()
    except Exception as exc:
        add("retrieval", "fail", reason=str(exc))
        return None

    queries = {
        "sunflowers": "hoa hướng dương",
        "roses": "hoa hồng",
        "daisy": "hoa cúc họa mi",
        "dandelion": "hoa bồ công anh",
        "tulips": "hoa tulip",
    }
    hits = 0
    details = {}
    for label, q in queries.items():
        rows = engine.search_text(q, 5)
        ok = any(label.rstrip('s') in str(r.get("label", "")).lower() or label in str(r.get("label", "")).lower() for r in rows)
        hits += int(ok)
        details[q] = ok
    rate = hits / len(queries)
    add("retrieval", "pass" if rate >= 0.8 else "warn", hit_at_5=round(rate, 3), details=details)
    return engine


def check_rag():
    try:
        from core.llm import Retriever, load_chunks
        retriever = Retriever(load_chunks())
    except Exception as exc:
        add("rag_retriever", "fail", reason=str(exc))
        return
    cases = {
        "Đổi trả trong bao lâu?": "doi_tra.md",
        "Phí giao hàng thế nào?": "giao_hang.md",
        "Có những cách thanh toán nào?": "thanh_toan.md",
        "Chính sách bảo hành?": "bao_hanh.md",
    }
    hits = 0
    details = {}
    for q, source in cases.items():
        rows = retriever.search(q, 3)
        got = [r["source"] for r in rows]
        ok = source in got
        hits += int(ok)
        details[q] = {"expected": source, "got": got}
    rate = hits / len(cases)
    add("rag_retriever", "pass" if rate >= 0.75 else "warn", source_hit_at_3=round(rate, 3), details=details)


def check_calorie(shared_retrieval=None, max_images=8):
    img_dir = DATA_DIR / "coco128" / "images" / "train2017"
    label_dir = DATA_DIR / "coco128" / "labels" / "train2017"
    if not img_dir.exists() or not label_dir.exists():
        add("calorie", "warn", reason="Thiếu COCO128; chỉ kiểm tra bằng tay trên ảnh món ăn.")
        return
    from core.calorie import FoodCalorieEstimator
    est = FoodCalorieEstimator()
    if shared_retrieval is not None:
        est.set_encoder(shared_retrieval.encoder)

    # Map lớp food của COCO sang nhãn kcal tương ứng.
    coco_map = {46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',51:'carrot',52:'hot_dog',53:'pizza',54:'donut',55:'cake'}
    samples = []
    for p in sorted(img_dir.glob("*.jpg")):
        lf = label_dir / (p.stem + ".txt")
        ids = {int(x.split()[0]) for x in lf.read_text().splitlines() if x.strip()} if lf.exists() else set()
        expected = [coco_map[i] for i in ids if i in coco_map]
        if expected:
            samples.append((p, expected))
        if len(samples) >= max_images:
            break
    if not samples:
        add("calorie", "warn", reason="COCO128 hiện không có sample food phù hợp để auto-test.")
        return
    hit = 0
    details = []
    for p, expected in samples:
        r = est.estimate(Image.open(p), grams=100, top_k=5)
        got = [x["label"] for x in r["top_predictions"]]
        ok = any(e in got for e in expected)
        hit += int(ok)
        details.append({"file": p.name, "expected": expected, "got": got, "ok": ok})
    rate = hit / len(samples)
    add("calorie", "pass" if rate >= 0.50 else "warn", top5_hit=round(rate, 3), samples=len(samples), details=details,
        note="CLIP food zero-shot là ước tính; UI cho phép người dùng xác nhận lại món.")


def main():
    print("=== VERIFY 5 AI ===")
    print("Project:", ROOT)
    check_files()
    try:
        check_classifier()
    except Exception as exc:
        add("classifier", "fail", reason=f"{type(exc).__name__}: {exc}")
    try:
        check_detector()
    except Exception as exc:
        add("detector", "fail", reason=f"{type(exc).__name__}: {exc}")
    engine = None
    try:
        engine = check_retrieval()
    except Exception as exc:
        add("retrieval", "fail", reason=f"{type(exc).__name__}: {exc}")
    try:
        check_rag()
    except Exception as exc:
        add("rag_retriever", "fail", reason=f"{type(exc).__name__}: {exc}")
    try:
        check_calorie(engine)
    except Exception as exc:
        add("calorie", "fail", reason=f"{type(exc).__name__}: {exc}")

    REPORT["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ART_DIR / "verification_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBáo cáo:", out)
    print("PASS = ổn; WARN = nên xem lại/đánh giá thủ công; FAIL = lỗi cần sửa.")


if __name__ == "__main__":
    main()
