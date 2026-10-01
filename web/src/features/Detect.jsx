import { useState } from 'react';
import { postImage } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Detect() {
  const [conf, setConf] = useState(0.25);
  const [file, setFile] = useState(null);
  const [state, setState] = useState({ status: 'idle' });

  async function run(f = file) {
    if (!f) return;
    setState({ status: 'loading' });
    try {
      setState({ status: 'ok', data: await postImage('/api/detect', f, { conf }) });
    } catch (err) {
      setState({ status: 'error', error: err.message });
    }
  }

  return (
    <section className="grid">
      <div>
        <h2>Phát hiện đối tượng</h2>
        <p className="muted">YOLO11n, 80 lớp COCO (người, xe, động vật, đồ vật…).</p>
        <label>Ngưỡng tin cậy: {conf.toFixed(2)}
          <input type="range" min="0.05" max="0.95" step="0.05" value={conf}
                 onChange={(e) => setConf(Number(e.target.value))} onMouseUp={() => run()} onTouchEnd={() => run()} />
        </label>
        <ImagePicker onChange={(f) => { setFile(f); run(f); }} />
      </div>
      <div>
        {state.status === 'loading' && <p>Đang phát hiện…</p>}
        {state.status === 'error' && <p className="error">{state.error}</p>}
        {state.status === 'ok' && (
          <>
            <img src={state.data.image} alt="Kết quả phát hiện" className="preview" />
            <p className="muted">{state.data.detections.length} đối tượng · {state.data.latency_ms} ms</p>
            <table>
              <thead><tr><th>Lớp</th><th>Độ tin cậy</th><th>Hộp (x1, y1, x2, y2)</th></tr></thead>
              <tbody>
                {state.data.detections.map((d, i) => (
                  <tr key={i}><td>{d.label}</td><td>{(d.score * 100).toFixed(1)}%</td><td>{d.box_xyxy.join(', ')}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </section>
  );
}