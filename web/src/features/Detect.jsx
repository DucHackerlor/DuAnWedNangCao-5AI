import { useState } from 'react';
import { postImage } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Detect({ disabled = false }) {
  const [conf, setConf] = useState(0.30);
  const [file, setFile] = useState(null);
  const [state, setState] = useState({ status: 'idle' });

  async function run() {
    if (!file || disabled || state.status === 'loading') return;
    setState({ status: 'loading' });
    try {
      setState({ status: 'ok', data: await postImage('/api/detect', file, { conf }) });
    } catch (err) {
      setState({ status: 'error', error: err.message });
    }
  }

  return (
    <section className="grid">
      <div>
        <h2>Phát hiện đối tượng</h2>
        <p className="muted">YOLO11s. Ca khó sẽ tự kiểm tra lại ở độ phân giải cao thay vì khẳng định quá sớm.</p>
        <label>Ngưỡng phát hiện: <b>{Math.round(conf * 100)}%</b>
          <input type="range" min="0.15" max="0.80" step="0.05" value={conf}
                 onChange={(e) => setConf(Number(e.target.value))} disabled={disabled} />
        </label>
        <ImagePicker disabled={disabled} onChange={(f) => { setFile(f); setState({ status: 'idle' }); }} />
        <button className="button primaryAction" onClick={run}
                disabled={!file || disabled || state.status === 'loading'}>
          {state.status === 'loading' ? 'Đang nhận diện…' : 'Nhận diện đối tượng'}
        </button>
        <p className="muted smallNote">Mặc định 30% giúp không bỏ sót; kết quả dưới 50% sẽ bị gắn cờ cần kiểm tra.</p>
      </div>
      <div>
        {state.status === 'idle' && <div className="emptyBox">Chọn ảnh rồi bấm <b>Nhận diện đối tượng</b>.</div>}
        {state.status === 'loading' && <div className="emptyBox">AI đang phân tích ảnh…</div>}
        {state.status === 'error' && <p className="error">{state.error}</p>}
        {state.status === 'ok' && (
          <>
            <img src={state.data.image} alt="Kết quả phát hiện" className="preview" />
            <div className="resultMeta">
              <span>{state.data.detections.length} đối tượng</span>
              <span>{state.data.latency_ms} ms</span>
              {state.data.used_retry && <span>Đã kiểm tra lại ca khó</span>}
            </div>
            {state.data.warnings?.map((w, i) => <p className="warn" key={i}>{w}</p>)}
            <table>
              <thead><tr><th>Đối tượng</th><th>Tin cậy</th><th>Mức</th><th>Hộp</th></tr></thead>
              <tbody>
                {state.data.detections.map((d, i) => (
                  <tr key={i} className={d.needs_review ? 'lowConfidenceRow' : ''}>
                    <td><b>{d.label_vi || d.label}</b><br/><small className="muted">{d.label_en || d.label}</small></td>
                    <td>{(d.score * 100).toFixed(1)}%</td>
                    <td>{d.confidence_level || '—'}</td>
                    <td>{d.box_xyxy.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </section>
  );
}
