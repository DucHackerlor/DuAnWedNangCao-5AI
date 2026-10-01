import { useState } from 'react';
import { postImage } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Classify() {
  const [state, setState] = useState({ status: 'idle' });

  async function run(file) {
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
        <p className="muted">ResNet-18 fine-tune trên 5 loài: daisy, dandelion, roses, sunflowers, tulips.</p>
        <ImagePicker onChange={run} />
      </div>
      <div>
        {state.status === 'loading' && <p>Đang dự đoán…</p>}
        {state.status === 'error' && <p className="error">{state.error}</p>}
        {state.status === 'ok' && (
          <>
            {!state.data.confident && <p className="warn">Mô hình không chắc chắn — ảnh có thể không thuộc 5 loài đã học.</p>}
            {state.data.predictions.map((p) => (
              <div key={p.label} className="bar">
                <span>{p.label}</span>
                <div className="track"><div className="fill" style={{ width: `${p.score * 100}%` }} /></div>
                <span>{(p.score * 100).toFixed(1)}%</span>
              </div>
            ))}
            <p className="muted">{state.data.latency_ms} ms</p>
          </>
        )}
      </div>
    </section>
  );
}