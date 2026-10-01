"""Giao diện Streamlit — client mỏng gọi FastAPI (mô hình chỉ nạp một lần ở backend)."""
import base64
import io
import json
import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Web Apps", page_icon="🤖", layout="wide")
API_URL = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def health(url: str):
    try:
        return requests.get(f"{url}/api/health", timeout=5).json()
    except requests.RequestException as exc:
        return {"status": "down", "error": str(exc), "models": {}}


h = health(API_URL)
st.sidebar.markdown(f"**Backend:** {'🟢 ' + h.get('device', '') if h['status'] == 'ok' else '🔴 không kết nối'}")
for name, ok in h.get("models", {}).items():
    st.sidebar.write(("✅ " if ok else "⛔ ") + name)


def post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Không gọi được API: {exc}")
        return None
    if not r.ok:
        st.error(f"Lỗi {r.status_code}: {r.json().get('detail', r.text) if r.headers.get('content-type', '').startswith('application/json') else r.text}")
        return None
    return r.json()


def upload(label: str, key: str):
    f = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key)
    if f:
        st.image(f, caption="Ảnh đầu vào", width="stretch")
    return f


st.title("🤖 AI Web Apps")
st.caption("Phân loại ảnh · Phát hiện đối tượng · Tìm kiếm ảnh · Chatbot RAG · Ước tính calo — một backend FastAPI")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌼 Phân loại", "🚗 Phát hiện", "🔎 Tìm ảnh", "💬 Chatbot", "🍽️ Đo calo"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh một bông hoa (daisy, dandelion, roses, sunflowers, tulips)", "cls")
        top_k = st.slider("Top-k", 1, 5, 3)
    if f and (res := post("/api/classify", files={"file": f.getvalue()}, data={"top_k": top_k})):
        with c2:
            if not res["confident"]:
                st.warning("Mô hình không chắc chắn — ảnh có thể không thuộc 5 loài đã học.")
            for p in res["predictions"]:
                st.progress(p["score"], text=f"{p['label']}: {p['score']:.1%}")
            st.caption(f"⏱ {res['latency_ms']} ms")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh bất kỳ (người, xe, động vật, đồ vật…)", "det")
        conf = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
    if f and (res := post("/api/detect", files={"file": f.getvalue()}, data={"conf": conf})):
        with c2:
            img = Image.open(io.BytesIO(base64.b64decode(res["image"].split(",", 1)[1])))
            st.image(img, caption=f"{len(res['detections'])} đối tượng · {res['latency_ms']} ms", width="stretch")
            st.write(res["summary"])
            st.dataframe(res["detections"], width="stretch")

with tab3:
    mode = st.radio("Tìm bằng", ["Câu mô tả (tiếng Anh)", "Ảnh mẫu"], horizontal=True)
    k = st.slider("Số kết quả", 4, 24, 8, 4)
    res = None
    if mode.startswith("Câu"):
        q = st.text_input("Ví dụ: a red flower, a dog on a sofa, people riding bikes", "yellow sunflowers in a field")
        if q:
            res = post("/api/search/text", json={"query": q, "k": k})
    else:
        f = upload("Ảnh mẫu", "ret")
        if f:
            res = post("/api/search/image", files={"file": f.getvalue()}, data={"k": k})
    if res:
        cols = st.columns(4)
        for i, r in enumerate(res["results"]):
            # tải ảnh phía server Streamlit: trình duyệt có thể không truy cập trực tiếp được API_URL
            img_bytes = requests.get(f"{API_URL}{r['url']}", timeout=30).content
            cols[i % 4].image(img_bytes, caption=f"{r['label']} · {r['score']:.3f}", width="stretch")

with tab4:
    st.info("Trợ lý ShopLite trả lời dựa trên tài liệu chính sách (RAG). Thử: *Đổi trả trong bao lâu?*")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    if prompt := st.chat_input("Nhập câu hỏi…"):
        st.chat_message("user").markdown(prompt)
        sources = []

        def stream():
            with requests.post(f"{API_URL}/api/chat", json={"message": prompt, "history": st.session_state.chat},
                               stream=True, timeout=300) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    if ev["type"] == "sources":
                        sources.extend(ev["items"])
                    elif ev["type"] == "token":
                        yield ev["text"]

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream())
            except requests.RequestException as exc:
                answer = f"Lỗi: {exc}"
                st.error(answer)
            with st.expander("Nguồn đã dùng"):
                for s in sources:
                    st.markdown(f"**{s['source']}** · điểm {s['score']}\n\n> {s['text'][:300]}…")
        st.session_state.chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]

with tab5:
    st.subheader("🍽️ AI ước tính calo món ăn")
    st.caption("Food-101 nhận diện món ăn; calo được ước tính theo số gram bạn nhập.")
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh món ăn", "calorie")
        grams = st.number_input("Khối lượng khẩu phần (gram)", min_value=1, max_value=3000, value=200, step=10)
        go = st.button("Ước tính calo", type="primary", disabled=not f)
    if go and f:
        with c2:
            with st.spinner("AI đang nhận diện món ăn… Lần đầu có thể cần tải model."):
                res = post("/api/calorie", files={"file": f.getvalue()}, data={"grams": grams, "top_k": 3})
            if res:
                st.metric("Calo ước tính", f"{res['estimated_kcal']} kcal")
                st.write(f"**Món ăn:** {res['food']['name']} ({res['food']['confidence']:.1%})")
                st.write(f"**Khẩu phần:** {res['portion_grams']} g")
                st.write(f"**Năng lượng tham khảo:** {res['kcal_per_100g']} kcal / 100g")
                lo, hi = res["estimated_range_kcal"]
                st.write(f"**Khoảng tham khảo:** {lo}–{hi} kcal")
                for p in res["top_predictions"]:
                    st.progress(p["confidence"], text=f"{p['name']}: {p['confidence']:.1%}")
                st.warning(res["warning"])
