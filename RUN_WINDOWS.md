# Cách chạy bản giao diện mới

## Terminal 1 — FastAPI
```bash
source ~/.venv/Scripts/activate
uvicorn api.main:app --reload
```
Chờ: `Application startup complete.`

## Terminal 2 — React (giao diện đẹp)
```bash
cd web
npm install
npm run dev
```
Mở địa chỉ Vite hiện ra, thường là `http://localhost:5173`.

## Terminal 3 — Streamlit (nếu muốn dùng giao diện Streamlit)
```bash
source ~/.venv/Scripts/activate
streamlit run streamlit_app.py
```

## Kiểm tra 4 AI
Mở `http://127.0.0.1:8000/api/health`.
Nếu model nào là `false`, xem trường `errors`.


## AI calo
Không cần cài thêm thư viện ngoài `requirements.txt`.
Lần đầu sử dụng cần Internet để tải model `nateraw/food`.

Sau khi backend và React đang chạy:
1. Chọn **Đo calo** ở sidebar.
2. Upload ảnh món ăn.
3. Nhập khối lượng gram.
4. Bấm **Ước tính calo**.

## Kiểm tra chất lượng trước khi demo

Sau khi đã có `data/flowers`, `data/coco128` và các artifact:

```bash
source ~/.venv/Scripts/activate
python verify_ai.py
```

Mở `artifacts/verification_report.json` để xem kết quả từng AI. `WARN` không nhất thiết là lỗi chương trình; nó nghĩa là model/dataset cần xem lại thủ công.

### Lưu ý YOLO11s
Bản Accuracy Fix mặc định dùng `yolo11s.pt` để tăng độ chính xác so với `yolo11n.pt`. Lần chạy đầu Ultralytics có thể tải weight mới. Nếu máy quá yếu có thể quay về model nhẹ bằng biến môi trường `YOLO_WEIGHTS=artifacts/detector/yolo11n.pt`.
