"""
Chuẩn bị 2 AI còn thiếu theo đúng notebook:
1) ResNet-18 phân loại 5 loài hoa -> artifacts/classifier/model.pt + classes.json
3) CLIP + FAISS tìm kiếm ảnh -> artifacts/retrieval/index.faiss + meta.json

Chạy file này tại THƯ MỤC GỐC project:
    python prepare_missing_ai.py
"""

from __future__ import annotations

import gc
import json
import random
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import config
from config import ART_DIR, DATA_DIR, DEVICE, ROOT

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FLOWERS_DIR = DATA_DIR / "flowers" / "flower_photos"
COCO_DIR = DATA_DIR / "coco128"
GALLERY = DATA_DIR / "gallery"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[OK] Đã có: {dest}")
        return dest

    print(f"[TẢI] {url}")
    print(f"      -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def prepare_datasets():
    print("\n=== BƯỚC A: CHUẨN BỊ DỮ LIỆU ===")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FLOWERS_DIR.exists():
        tgz = download(
            "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            DATA_DIR / "flower_photos.tgz",
        )
        print("[GIẢI NÉN] TF Flowers...")
        with tarfile.open(tgz) as t:
            try:
                t.extractall(DATA_DIR / "flowers", filter="data")
            except TypeError:
                t.extractall(DATA_DIR / "flowers")
        tgz.unlink(missing_ok=True)

    (FLOWERS_DIR / "LICENSE.txt").unlink(missing_ok=True)

    if not COCO_DIR.exists():
        z = download(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
            DATA_DIR / "coco128.zip",
        )
        print("[GIẢI NÉN] COCO128...")
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_DIR)
        z.unlink(missing_ok=True)

    flower_counts = {
        d.name: len(list(d.glob("*.jpg")))
        for d in sorted(FLOWERS_DIR.iterdir())
        if d.is_dir()
    }
    coco_images = sorted((COCO_DIR / "images" / "train2017").glob("*.jpg"))

    print("[OK] Flowers:", flower_counts)
    print("[OK] COCO128:", len(coco_images), "ảnh")
    return coco_images


def train_classifier():
    print("\n=== BƯỚC B: HUẤN LUYỆN PHÂN LOẠI HOA (RESNET-18, ACCURACY MODE) ===")
    print("Thiết bị:", DEVICE)

    import core.classifier as clf

    base = ImageFolder(FLOWERS_DIR)
    classes = base.classes
    targets = np.array(base.targets)
    all_idx = np.arange(len(targets))

    train_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=targets[tmp_idx], random_state=SEED)

    # Không còn FAST mode 800 ảnh/1 epoch. Dùng toàn bộ train split để giảm sai số.
    head_epochs = 2
    finetune_epochs = 2 if DEVICE == "cpu" else 4

    train_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.TRAIN_TF), train_idx)
    val_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.EVAL_TF), val_idx)
    test_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.EVAL_TF), test_idx)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0, pin_memory=(DEVICE == "cuda"))

    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    out = ART_DIR / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({
        "train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()
    }), encoding="utf-8")

    print(f"Lớp: {classes}\ntrain={len(train_ds)} · val={len(val_ds)} · test={len(test_ds)}")

    model = clf.build_model(len(classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_epoch(dl, train, optimizer=None):
        model.train(train)
        total = correct = 0
        loss_sum = 0.0
        for step, (x, y) in enumerate(dl, 1):
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            with torch.set_grad_enabled(train), torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                logits = model(x); loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item(); total += len(y)
            if train and step % 10 == 0:
                print(f"  batch {step}/{len(dl)}")
        return loss_sum / total, correct / total

    best_acc = -1.0
    history = []

    # Phase 1: chỉ học classifier head để ổn định nhanh.
    for param in model.parameters(): param.requires_grad = False
    for param in model.fc.parameters(): param.requires_grad = True
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    epoch_no = 0
    for _ in range(head_epochs):
        epoch_no += 1; t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_dl, True, optimizer)
        va_loss, va_acc = run_epoch(val_dl, False)
        history.append({"epoch": epoch_no, "phase": "head", "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_acc:
            best_acc = va_acc; torch.save(model.state_dict(), out / "model.pt")
        print(f"[HEAD {epoch_no}] train={tr_acc:.3f} · val={va_acc:.3f} · {time.time()-t0:.0f}s")

    # Phase 2: fine-tune layer4 + fc với learning rate thấp.
    for param in model.layer4.parameters(): param.requires_grad = True
    for param in model.fc.parameters(): param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [{"params": model.layer4.parameters(), "lr": 8e-5}, {"params": model.fc.parameters(), "lr": 2e-4}],
        weight_decay=1e-4,
    )

    stale = 0
    for _ in range(finetune_epochs):
        epoch_no += 1; t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_dl, True, optimizer)
        va_loss, va_acc = run_epoch(val_dl, False)
        history.append({"epoch": epoch_no, "phase": "finetune", "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_acc + 1e-4:
            best_acc = va_acc; stale = 0; torch.save(model.state_dict(), out / "model.pt")
        else:
            stale += 1
        print(f"[FT {epoch_no}] train={tr_acc:.3f} · val={va_acc:.3f} · {time.time()-t0:.0f}s")
        if stale >= 2:
            print("Early stopping: val accuracy không cải thiện 2 epoch.")
            break

    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    model.load_state_dict(torch.load(out / "model.pt", map_location=DEVICE, weights_only=True)); model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in test_dl:
            y_pred += model(x.to(DEVICE)).argmax(1).cpu().tolist(); y_true += y.tolist()

    metrics = {
        "test_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "epochs": len(history), "history": history,
        "model": "resnet18-imagenet-two-stage-finetune",
        "train_samples": len(train_ds),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK]", out / "model.pt")
    print(f"[KẾT QUẢ] test accuracy={metrics['test_accuracy']:.3f} · macro-F1={metrics['test_macro_f1']:.3f}")

    del model, train_dl, val_dl, test_dl
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def build_retrieval(coco_images):
    print("\n=== BƯỚC C: TẠO AI TÌM KIẾM ẢNH (CLIP + FAISS) ===")

    import core.retrieval as ret
    from core.detector import ObjectDetector

    GALLERY.mkdir(parents=True, exist_ok=True)

    # Xóa gallery cũ để meta và dữ liệu luôn đồng bộ.
    for p in GALLERY.iterdir():
        if p.is_file():
            p.unlink()

    print("[1/3] Gắn nhãn 128 ảnh COCO bằng YOLO...")
    detector = ObjectDetector()

    items = []
    for idx, p in enumerate(coco_images, 1):
        summary = detector.detect(Image.open(p), conf=0.4)[0]["summary"]
        label = ", ".join(
            sorted(summary, key=summary.get, reverse=True)[:2]
        ) or "coco"

        dst = GALLERY / f"coco_{p.name}"
        shutil.copy2(p, dst)
        items.append(
            {
                "path": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "label": label,
                "source": "coco128",
            }
        )

        if idx % 20 == 0 or idx == len(coco_images):
            print(f"  COCO {idx}/{len(coco_images)}")

    print("[2/3] Thêm tối đa 100 ảnh cho mỗi loài hoa...")
    rng = random.Random(SEED)
    classes = sorted(d.name for d in FLOWERS_DIR.iterdir() if d.is_dir())

    for c in classes:
        files = sorted((FLOWERS_DIR / c).glob("*.jpg"))
        chosen = rng.sample(files, min(100, len(files)))

        for p in chosen:
            dst = GALLERY / f"{c}_{p.name}"
            shutil.copy2(p, dst)
            items.append(
                {
                    "path": str(dst.relative_to(ROOT)).replace("\\", "/"),
                    "label": c,
                    "source": "flowers",
                }
            )

    print(f"[OK] Kho ảnh có {len(items)} ảnh.")

    print("[3/3] Đang mã hóa ảnh bằng CLIP và lập FAISS index...")
    print("      Lần đầu có thể tải model CLIP từ Hugging Face.")
    encoder = ret.ClipEncoder()

    t0 = time.time()
    ret.build_index(encoder, items)
    engine = ret.ImageSearch(encoder=encoder)

    print(
        f"[OK] Đã lập chỉ mục {engine.index.ntotal} ảnh "
        f"trong {time.time() - t0:.0f}s"
    )
    print("[OK] Đã tạo:")
    print("    ", ART_DIR / "retrieval" / "index.faiss")
    print("    ", ART_DIR / "retrieval" / "meta.json")

    # Smoke test giống notebook.
    result = engine.search_text("yellow sunflowers in a field", k=3)
    print("[TEST] 'yellow sunflowers in a field':")
    for r in result:
        print(f"       {r['label']} · score={r['score']}")

    del detector, encoder, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify():
    print("\n=== KIỂM TRA FILE CUỐI CÙNG ===")
    required = [
        ART_DIR / "classifier" / "model.pt",
        ART_DIR / "classifier" / "classes.json",
        ART_DIR / "retrieval" / "index.faiss",
        ART_DIR / "retrieval" / "meta.json",
    ]

    all_ok = True
    for p in required:
        ok = p.exists() and p.stat().st_size > 0
        print(("✅" if ok else "❌"), p.relative_to(ROOT))
        all_ok &= ok

    if all_ok:
        print("\n🎉 XONG. 2 AI còn thiếu đã có đủ artifact.")
        print("Bây giờ chạy lại backend:")
        print("  uvicorn api.main:app --reload")
        print("Sau đó mở:")
        print("  http://127.0.0.1:8000/api/health")
        print("Mục tiêu: classifier=true, detector=true, retrieval=true, llm=true")
    else:
        print("\nCó file chưa tạo được. Hãy gửi ảnh lỗi Terminal để kiểm tra.")


def main():
    print("Project:", ROOT)
    print("Python/Torch device:", DEVICE)
    print(
        "\nLƯU Ý: Nếu backend uvicorn đang chạy, hãy Ctrl+C trước khi chạy script này "
        "để tiết kiệm RAM.\n"
    )

    coco_images = prepare_datasets()
    train_classifier()
    build_retrieval(coco_images)
    verify()


if __name__ == "__main__":
    main()
