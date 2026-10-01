import { useEffect, useMemo, useState } from 'react';
import { getHealth } from './api.js';
import Classify from './features/Classify.jsx';
import Detect from './features/Detect.jsx';
import Search from './features/Search.jsx';
import Chat from './features/Chat.jsx';
import Calories from './features/Calories.jsx';

const TABS = [
  { id: 'classify', icon: '✿', label: 'Phân loại hoa', model: 'classifier', desc: 'ResNet-18 + CLIP', Component: Classify },
  { id: 'detect', icon: '⌖', label: 'Nhận diện', model: 'detector', desc: 'YOLO11s + retry', Component: Detect },
  { id: 'search', icon: '⌕', label: 'Tìm kiếm ảnh', model: 'retrieval', desc: 'CLIP + FAISS', Component: Search },
  { id: 'chat', icon: '✦', label: 'Trợ lý AI', model: 'llm', desc: 'RAG + Qwen', Component: Chat },
  { id: 'calorie', icon: '◉', label: 'Đo calo', model: 'calorie', desc: 'CLIP food + kcal', Component: Calories },
];

export default function App() {
  const [tab, setTab] = useState('detect');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    const refresh = () => getHealth()
      .then((data) => active && setHealth(data))
      .catch(() => active && setHealth({ status: 'down', models: {}, errors: {} }));
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  const current = useMemo(() => TABS.find((t) => t.id === tab), [tab]);
  const Current = current.Component;
  const ready = Boolean(health?.models?.[current.model]);
  const readyCount = Object.values(health?.models ?? {}).filter(Boolean).length;

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandLogo">AI</div>
          <div><strong>AI Studio</strong><small>Web nâng cao</small></div>
        </div>

        <div className="backendCard">
          <div className="backendLine">
            <span className={`statusDot ${health?.status === 'ok' ? 'on' : ''}`} />
            <b>{health?.status === 'ok' ? 'Backend online' : 'Backend offline'}</b>
          </div>
          <span>{health?.status === 'ok' ? `${readyCount}/5 AI sẵn sàng · ${String(health.device).toUpperCase()}` : 'Kiểm tra FastAPI cổng 8000'}</span>
        </div>

        <nav className="navList">
          {TABS.map((item) => {
            const ok = Boolean(health?.models?.[item.model]);
            return (
              <button key={item.id} className={`navItem ${tab === item.id ? 'active' : ''}`} onClick={() => setTab(item.id)}>
                <span className="navIcon">{item.icon}</span>
                <span className="navCopy"><b>{item.label}</b><small>{item.desc}</small></span>
                <span className={`modelDot ${ok ? 'ready' : ''}`} />
              </button>
            );
          })}
        </nav>
        <div className="sidebarFoot">FastAPI · React · Streamlit</div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">DỰ ÁN WEB TÍCH HỢP 5 AI</span>
            <h1>{current.label}</h1>
            <p>{current.desc} · xử lý tập trung qua FastAPI</p>
          </div>
          <div className={`readyBadge ${ready ? 'ok' : 'bad'}`}>{ready ? '● AI sẵn sàng' : '● Chưa sẵn sàng'}</div>
        </header>

        <main className="mainContent">
          {!ready && health && (
            <div className="modelNotice">
              <b>{current.label} hiện chưa sử dụng được.</b>
              <span>{health.errors?.[current.model] || 'Model chưa được nạp. Mở /api/health để kiểm tra.'}</span>
            </div>
          )}
          <div className="heroCard">
            <div className="heroGlow" />
            <div>
              <span className="heroKicker">SMART AI WORKSPACE</span>
              <h2>Một giao diện gọn gàng cho 5 mô hình AI</h2>
              <p>Ảnh, tìm kiếm ngữ nghĩa, chatbot và ước tính calo được xử lý trên cùng một backend để giảm tải và phản hồi ổn định hơn.</p>
            </div>
            <div className="heroStats">
              <div><b>{readyCount}</b><span>AI online</span></div>
              <div><b>{health?.device ? String(health.device).toUpperCase() : '—'}</b><span>Thiết bị</span></div>
            </div>
          </div>
          <section className="featureCard"><Current disabled={!ready} /></section>
        </main>
      </div>
    </div>
  );
}
