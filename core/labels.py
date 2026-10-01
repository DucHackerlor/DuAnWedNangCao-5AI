"""Các hàm thuần dùng chung: dịch nhãn COCO, chuẩn hoá truy vấn và đánh giá độ chắc chắn."""
from __future__ import annotations

import re

COCO_VI = {
    "person": "người", "bicycle": "xe đạp", "car": "ô tô", "motorcycle": "xe máy",
    "airplane": "máy bay", "bus": "xe buýt", "train": "tàu hỏa", "truck": "xe tải",
    "boat": "thuyền", "traffic light": "đèn giao thông", "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển dừng", "parking meter": "đồng hồ đỗ xe", "bench": "ghế băng",
    "bird": "chim", "cat": "mèo", "dog": "chó", "horse": "ngựa", "sheep": "cừu",
    "cow": "bò", "elephant": "voi", "bear": "gấu", "zebra": "ngựa vằn", "giraffe": "hươu cao cổ",
    "backpack": "ba lô", "umbrella": "ô", "handbag": "túi xách", "tie": "cà vạt",
    "suitcase": "va li", "frisbee": "đĩa bay", "skis": "ván trượt tuyết", "snowboard": "ván trượt",
    "sports ball": "bóng thể thao", "kite": "diều", "baseball bat": "gậy bóng chày",
    "baseball glove": "găng bóng chày", "skateboard": "ván trượt", "surfboard": "ván lướt sóng",
    "tennis racket": "vợt tennis", "bottle": "chai", "wine glass": "ly rượu", "cup": "cốc",
    "fork": "nĩa", "knife": "dao", "spoon": "thìa", "bowl": "bát", "banana": "chuối",
    "apple": "táo", "sandwich": "sandwich", "orange": "cam", "broccoli": "bông cải xanh",
    "carrot": "cà rốt", "hot dog": "hot dog", "pizza": "pizza", "donut": "donut", "cake": "bánh ngọt",
    "chair": "ghế", "couch": "ghế sofa", "potted plant": "chậu cây", "bed": "giường",
    "dining table": "bàn ăn", "toilet": "bồn cầu", "tv": "tivi", "laptop": "laptop",
    "mouse": "chuột máy tính", "remote": "điều khiển", "keyboard": "bàn phím", "cell phone": "điện thoại",
    "microwave": "lò vi sóng", "oven": "lò nướng", "toaster": "máy nướng bánh", "sink": "bồn rửa",
    "refrigerator": "tủ lạnh", "book": "sách", "clock": "đồng hồ", "vase": "bình hoa",
    "scissors": "kéo", "teddy bear": "gấu bông", "hair drier": "máy sấy tóc", "toothbrush": "bàn chải đánh răng",
}

FLOWER_VI = {
    "daisy": "hoa cúc họa mi",
    "dandelion": "hoa bồ công anh",
    "roses": "hoa hồng",
    "rose": "hoa hồng",
    "sunflowers": "hoa hướng dương",
    "sunflower": "hoa hướng dương",
    "tulips": "hoa tulip",
    "tulip": "hoa tulip",
}

# Những từ thường dùng trong giao diện tiếng Việt. Đây không phải dịch máy tổng quát;
# nó chỉ giúp CLIP tiếng Anh hiểu tốt hơn các truy vấn phổ biến trong bài demo.
VI_TO_EN_PHRASES = {
    "hoa hướng dương": "sunflower",
    "hoa bồ công anh": "dandelion",
    "hoa cúc họa mi": "daisy flower",
    "hoa tulip": "tulip flower",
    "hoa hồng": "rose flower",
    "con chó": "dog", "chó": "dog",
    "con mèo": "cat", "mèo": "cat",
    "con ngựa": "horse", "ngựa": "horse",
    "con bò": "cow", "bò": "cow",
    "con chim": "bird", "chim": "bird",
    "người": "person",
    "xe máy": "motorcycle", "ô tô": "car", "xe hơi": "car", "xe đạp": "bicycle",
    "xe buýt": "bus", "xe tải": "truck",
    "cua": "crab food", "tôm": "shrimp food", "tôm hùm": "lobster food",
    "cá": "fish food", "pizza": "pizza", "bánh": "cake", "đồ ăn": "food", "thức ăn": "food",
}


def coco_vi(label: str) -> str:
    return COCO_VI.get(label, label)


def flower_vi(label: str) -> str:
    return FLOWER_VI.get(label, label.replace("_", " "))


def confidence_band(score: float) -> str:
    if score >= 0.75:
        return "cao"
    if score >= 0.50:
        return "trung bình"
    return "thấp"


def normalize_search_query(query: str) -> str:
    """Dịch nhẹ các cụm Việt phổ biến sang tiếng Anh cho CLIP OpenAI."""
    out = " ".join(str(query).strip().split())
    low = out.lower()
    # cụm dài trước để tránh 'tôm' thay vào 'tôm hùm'.
    for vi, en in sorted(VI_TO_EN_PHRASES.items(), key=lambda kv: -len(kv[0])):
        low = re.sub(rf"(?<!\w){re.escape(vi)}(?!\w)", en, low, flags=re.IGNORECASE)
    return low


def token_overlap(query: str, label: str) -> int:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    l = set(re.findall(r"[a-z0-9]+", label.lower()))
    return len(q & l)
