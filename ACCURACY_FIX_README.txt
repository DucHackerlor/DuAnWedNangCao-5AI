BẢN SỬA ĐỘ CHÍNH XÁC 5 AI

1) Copy toàn bộ patch vào project hiện tại và chọn Replace.
2) Dừng backend cũ (Ctrl+C).
3) source ~/.venv/Scripts/activate
4) uvicorn api.main:app --reload
5) Terminal khác: cd web && npm run dev

LƯU Ý:
- YOLO mặc định đổi từ yolo11n sang yolo11s để tăng độ chính xác; lần đầu có thể tải weight mới.
- Phân loại hoa dùng thêm CLIP kiểm chứng nếu AI tìm kiếm ảnh đang hoạt động.
- Chatbot sẽ từ chối câu ngoài tài liệu thay vì đoán.
- AI calo không còn ép mọi ảnh thành một món; có kiểm tra food/non-food và cho phép xác nhận món.

TEST:
- ENABLED_MODELS= python -m pytest -q
- python verify_ai.py
