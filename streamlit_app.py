"""Giao diện Streamlit — client mỏng gọi FastAPI, đồng bộ với bản kiểm tra độ chính xác."""
import base64
import io
import json
import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Web Apps", page_icon="🤖", layout="wide")
API_URL = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")


@st.cache_data(ttl=15, show_spinner=False)
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
        r = requests.post(f"{API_URL}{path}", timeout=300, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Không gọi được API: {exc}")
        return None
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        st.error(f"Lỗi {r.status_code}: {detail}")
        return None
    return r.json()


def upload(label: str, key: str):
    f = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key)
    if f:
        st.image(f, caption="Ảnh đầu vào", width="stretch")
    return f


st.title("🤖 AI Web Apps — Accuracy Fix")
st.caption("5 AI có kiểm tra độ chắc chắn, tránh ép model phải trả lời khi bằng chứng yếu.")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌼 Phân loại", "🎯 Phát hiện", "🔎 Tìm ảnh", "💬 Chatbot", "🍽️ Đo calo"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh hoa: daisy, dandelion, roses, sunflowers, tulips", "cls")
        go = st.button("Phân loại", type="primary", disabled=not f)
    if go and f and (res := post("/api/classify", files={"file": f.getvalue()}, data={"top_k": 3})):
        with c2:
            if not res["confident"]:
                st.warning(res.get("reason") or "Mô hình chưa đủ chắc chắn.")
            if res.get("agreement") is False:
                st.warning("ResNet và CLIP chưa đồng thuận.")
            for p in res["predictions"]:
                st.progress(p["score"], text=f"{p.get('label_vi', p['label'])}: {p['score']:.1%}")
            st.caption(f"{res.get('method')} · ⏱ {res['latency_ms']} ms")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh bất kỳ (người, xe, động vật, đồ vật…)", "det")
        conf = st.slider("Ngưỡng phát hiện", 0.15, 0.80, 0.30, 0.05)
        go = st.button("Nhận diện", type="primary", disabled=not f)
    if go and f and (res := post("/api/detect", files={"file": f.getvalue()}, data={"conf": conf})):
        with c2:
            img = Image.open(io.BytesIO(base64.b64decode(res["image"].split(",", 1)[1])))
            st.image(img, caption=f"{len(res['detections'])} đối tượng · {res['latency_ms']} ms", width="stretch")
            for w in res.get("warnings", []):
                st.warning(w)
            st.write(res["summary"])
            st.dataframe(res["detections"], width="stretch")

with tab3:
    mode = st.radio("Tìm bằng", ["Câu mô tả", "Ảnh mẫu"], horizontal=True)
    k = st.slider("Số kết quả", 4, 24, 8, 4)
    res = None
    if mode == "Câu mô tả":
        with st.form("search_form"):
            q = st.text_input("Có thể nhập một số từ tiếng Việt phổ biến", "chó và mèo")
            go = st.form_submit_button("Tìm")
        if go and q:
            res = post("/api/search/text", json={"query": q, "k": k})
            if res and res.get("query_used") != q.lower():
                st.caption(f"Truy vấn CLIP: {res['query_used']}")
    else:
        f = upload("Ảnh mẫu", "ret")
        go = st.button("Tìm ảnh tương tự", disabled=not f)
        if go and f:
            res = post("/api/search/image", files={"file": f.getvalue()}, data={"k": k})
    if res:
        if res.get("warning"):
            st.warning(res["warning"])
        cols = st.columns(4)
        for i, r in enumerate(res.get("results", [])):
            img_bytes = requests.get(f"{API_URL}{r['url']}", timeout=30).content
            cols[i % 4].image(img_bytes, caption=f"{r['label']} · {r['score']:.3f} · {r.get('quality','')}", width="stretch")

with tab4:
    st.info("RAG chỉ trả lời khi tài liệu nội bộ đủ liên quan; câu ngoài phạm vi sẽ bị từ chối.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    if prompt := st.chat_input("Nhập câu hỏi…"):
        st.chat_message("user").markdown(prompt)
        sources = []

        def stream():
            with requests.post(f"{API_URL}/api/chat", json={"message": prompt, "history": st.session_state.chat}, stream=True, timeout=300) as r:
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
            if sources:
                with st.expander("Nguồn đã dùng"):
                    for s in sources:
                        st.markdown(f"**{s['source']}** · điểm {s['score']}\n\n> {s['text'][:300]}…")
        st.session_state.chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]

with tab5:
    st.subheader("🍽️ AI ước tính calo món ăn")
    st.caption("CLIP nhận diện nhóm món; nếu không chắc, hệ thống yêu cầu bạn xác nhận thay vì khẳng định sai.")
    c1, c2 = st.columns(2)
    with c1:
        f = upload("Ảnh món ăn", "calorie")
        grams = st.number_input("Khối lượng khẩu phần (gram)", min_value=1, max_value=3000, value=200, step=10)
        go = st.button("Ước tính calo", type="primary", disabled=not f)
    if go and f:
        with c2:
            with st.spinner("AI đang phân tích món ăn…"):
                res = post("/api/calorie", files={"file": f.getvalue()}, data={"grams": grams, "top_k": 5})
            if res:
                if not res.get("is_food"):
                    st.warning("AI chưa xác nhận đây là ảnh thức ăn nên không tính calo.")
                else:
                    st.metric("Calo ước tính", f"{res['estimated_kcal']} kcal")
                    st.write(f"**Món:** {res['food']['name']} · điểm phù hợp {res['food']['match_score']:.1%}")
                    if res.get("uncertain"):
                        st.warning("AI chưa chắc chắn; nên xác nhận món đúng.")
                    for p in res["top_predictions"]:
                        st.progress(p["match_score"], text=f"{p['name']}: {p['match_score']:.1%}")
                    options = {x["name"]: x["label"] for x in res["correction_options"]}
                    chosen_name = st.selectbox("Nếu AI nhận sai, chọn lại món", list(options))
                    if st.button("Tính lại theo món đã chọn"):
                        calc = post("/api/calorie/recalculate", json={"food_label": options[chosen_name], "grams": grams})
                        if calc:
                            st.metric("Calo sau xác nhận", f"{calc['estimated_kcal']} kcal")
                    st.warning(res["warning"])
