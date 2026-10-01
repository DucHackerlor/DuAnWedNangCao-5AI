import { useState } from 'react';
import { postImage } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Classify({ disabled = false }) {
  const [file, setFile] = useState(null);
  const [state, setState] = useState({ status: 'idle' });

  async function run() {
    if (!file || disabled || state.status === 'loading') return;
    setState({ status: 'loading' });
    try {
      setState({ status: 'ok', data: await postImage('/api/classify', file, { top_k: 3 }) });
    } catch (err) {
      setState({ status: 'error', error: err.message });
    }
  }

  return (
    <section className="grid">
      <div>
        <h2>Phân loại hoa</h2>
        <p className="muted">ResNet-18 được CLIP kiểm chứng để giảm việc ép ảnh không phải hoa vào một trong 5 lớp.</p>
        <ImagePicker disabled={disabled} label="Chọn ảnh hoa" onChange={(f) => { setFile(f); setState({ status: 'idle' }); }} />
        <button className="button primaryAction" onClick={run}
                disabled={!file || disabled || state.status === 'loading'}>
          {state.status === 'loading' ? 'Đang phân loại…' : 'Phân loại ảnh'}
        </button>
      </div>
      <div>
        {state.status === 'idle' && <div className="emptyBox">Kết quả sẽ xuất hiện ở đây.</div>}
        {state.status === 'loading' && <div className="emptyBox">ResNet + CLIP đang kiểm tra…</div>}
        {state.status === 'error' && <p className="error">{state.error}</p>}
        {state.status === 'ok' && (
          <>
            {!state.data.confident && <p className="warn">Không nên khẳng định: {state.data.reason}</p>}
            {state.data.agreement === false && <p className="warn">Hai mô hình chưa đồng thuận; hãy xem các dự đoán bên dưới.</p>}
            {state.data.predictions.map((p) => (
              <div key={p.label} className="bar">
                <span><b>{p.label_vi || p.label}</b></span>
                <div className="track"><div className="fill" style={{ width: `${p.score * 100}%` }} /></div>
                <span>{(p.score * 100).toFixed(1)}%</span>
              </div>
            ))}
            <p className="muted smallNote">{state.data.method} · margin {(state.data.margin * 100).toFixed(1)}% · {state.data.latency_ms} ms</p>
          </>
        )}
      </div>
    </section>
  );
}
