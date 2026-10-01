import { useState } from 'react';
import { API_BASE, postImage, postJson } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Search({ disabled = false }) {
  const [query, setQuery] = useState('chó và mèo');
  const [image, setImage] = useState(null);
  const [mode, setMode] = useState('text');
  const [state, setState] = useState({ status: 'idle' });

  async function runText(e) {
    e?.preventDefault();
    if (!query.trim() || disabled) return;
    setState({ status: 'loading' });
    try { setState({ status: 'ok', ...(await postJson('/api/search/text', { query, k: 12 })) }); }
    catch (err) { setState({ status: 'error', error: err.message }); }
  }

  async function runImage() {
    if (!image || disabled) return;
    setState({ status: 'loading' });
    try { setState({ status: 'ok', ...(await postImage('/api/search/image', image, { k: 12 })) }); }
    catch (err) { setState({ status: 'error', error: err.message }); }
  }

  return (
    <section>
      <h2>Tìm kiếm ảnh bằng CLIP</h2>
      <p className="muted">Hỗ trợ một số từ khóa tiếng Việt phổ biến và lọc bớt kết quả có độ liên quan quá thấp.</p>
      <div className="modeTabs">
        <button className={mode === 'text' ? 'active' : ''} onClick={() => setMode('text')}>Bằng mô tả</button>
        <button className={mode === 'image' ? 'active' : ''} onClick={() => setMode('image')}>Bằng ảnh</button>
      </div>
      {mode === 'text' ? (
        <form className="row" onSubmit={runText}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ví dụ: chó và mèo" aria-label="Câu mô tả" disabled={disabled} />
          <button className="button" type="submit" disabled={disabled}>Tìm</button>
        </form>
      ) : (
        <div>
          <ImagePicker disabled={disabled} label="Chọn ảnh mẫu" onChange={setImage} />
          <button className="button primaryAction" onClick={runImage} disabled={!image || disabled}>Tìm ảnh tương tự</button>
        </div>
      )}

      {state.status === 'loading' && <div className="emptyBox">Đang tìm kết quả phù hợp…</div>}
      {state.status === 'error' && <p className="error">{state.error}</p>}
      {state.status === 'ok' && state.warning && <p className="warn">{state.warning}</p>}
      {state.status === 'ok' && state.query_used && state.query_used !== query.toLowerCase() && (
        <p className="muted smallNote">CLIP hiểu truy vấn dưới dạng: <b>{state.query_used}</b></p>
      )}
      {state.status === 'ok' && (
        <div className="gallery">
          {(state.results || []).map((r) => (
            <figure key={r.id}>
              <img src={`${API_BASE}${r.url}`} alt={r.label} loading="lazy" />
              <figcaption><b>{r.label}</b><br/>score {r.score.toFixed(3)} · {r.quality || '—'}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </section>
  );
}
