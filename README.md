# AI Web Apps — Streamlit & React

Năm ứng dụng AI (phân loại ảnh, phát hiện đối tượng, tìm kiếm ảnh, chatbot RAG, ước tính calo món ăn) sau một backend FastAPI,
với hai giao diện: Streamlit và React.

## Chạy trên máy (Python 3.11, Node 22)
```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000                  # backend + React build (nếu có web/dist)
API_URL=http://localhost:8000 streamlit run streamlit_app.py
cd web && npm install && npm run dev              # React dev server, proxy /api → 8000
python -m pytest -q tests                         # test không cần GPU
```

## Biến môi trường
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ENABLED_MODELS` | `classifier,detector,retrieval,llm,calorie` | Mô hình được nạp |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` (GPU) / `-0.5B-` (CPU) | Mô hình sinh |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding cho RAG |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Tìm kiếm ảnh |
| `FOOD_MODEL` | `nateraw/food` | Nhận diện món ăn Food-101 |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8501` | Origin được gọi API |
| `API_URL` | `http://localhost:8000` | (Streamlit) địa chỉ backend |

## Docker
```bash
docker build -t ai-web-apps . && docker run -p 7860:7860 ai-web-apps   # mở http://localhost:7860
```

Chỉ số mô hình: xem `artifacts/*/metrics.json` và `artifacts/rag_metrics.json`.

## AI thứ 5 — Ước tính calo món ăn
- Model nhận diện: `nateraw/food` (ViT fine-tune trên Food-101).
- Lần đầu bấm **Ước tính calo**, model sẽ được tải từ Hugging Face và lưu vào cache máy.
- Người dùng nhập khối lượng (gram); hệ thống dùng kcal/100g tham khảo để tính tổng năng lượng.
- Kết quả chỉ phục vụ ước tính/demo vì dầu, sốt, công thức và khẩu phần thực tế có thể khác.

Model Food-101 được nạp lazy để 4 AI cũ không bị chậm khi backend khởi động.
