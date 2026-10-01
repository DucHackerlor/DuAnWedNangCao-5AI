import { useMemo, useState } from 'react';
import { postImage, postJson } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Calories({ disabled = false }) {
  const [file, setFile] = useState(null);
  const [grams, setGrams] = useState(200);
  const [state, setState] = useState({ status: 'idle' });
  const [manualLabel, setManualLabel] = useState('');

  async function run() {
    if (!file || disabled || state.status === 'loading') return;
    setState({ status: 'loading' });
    setManualLabel('');
    try {
      const data = await postImage('/api/calorie', file, { grams, top_k: 5 });
      setState({ status: 'ok', data });
      setManualLabel(data.food?.label || '');
    } catch (err) {
      setState({ status: 'error', error: err.message });
    }
  }

  async function recalculate(label) {
    setManualLabel(label);
    if (!label) return;
    try {
      const calc = await postJson('/api/calorie/recalculate', { food_label: label, grams });
      setState((s) => ({
        ...s,
        data: {
          ...s.data,
          food: { ...s.data.food, label: calc.food.label, name: calc.food.name },
          kcal_per_100g: calc.kcal_per_100g,
          estimated_kcal: calc.estimated_kcal,
          estimated_range_kcal: calc.estimated_range_kcal,
          portion_grams: calc.portion_grams,
        },
      }));
    } catch (err) {
      setState({ status: 'error', error: err.message });
    }
  }

  const selectedOption = useMemo(() => {
    if (state.status !== 'ok') return null;
    return state.data.correction_options?.find((x) => x.label === manualLabel) || null;
  }, [state, manualLabel]);

  return (
    <section className="grid calorieGrid">
      <div>
        <h2>AI ước tính calo món ăn</h2>
        <p className="muted">CLIP nhận diện nhóm món ăn, kiểm tra food/non-food và cho phép bạn xác nhận lại món trước khi dùng số calo.</p>
        <ImagePicker disabled={disabled} label="Chọn ảnh món ăn" onChange={(f) => { setFile(f); setState({ status: 'idle' }); }} />

        <div className="calorieControls">
          <label>Khối lượng khẩu phần <b>{grams} g</b></label>
          <input type="range" min="20" max="1000" step="10" value={grams}
                 onChange={(e) => setGrams(Number(e.target.value))} disabled={disabled} />
          <input className="gramsInput" type="number" min="1" max="3000" value={grams}
                 onChange={(e) => setGrams(Math.max(1, Math.min(3000, Number(e.target.value) || 1)))} disabled={disabled} />
        </div>

        <button type="button" className="calorieButton"
                disabled={!file || disabled || state.status === 'loading'} onClick={run}>
          {state.status === 'loading' ? 'AI đang phân tích…' : 'Ước tính calo'}
        </button>
        <p className="calorieHint">Không thể suy ra chính xác dầu, sốt và khối lượng chỉ từ một ảnh; kết quả là tham khảo.</p>
      </div>

      <div>
        {state.status === 'idle' && <div className="calorieEmpty"><span>🍽️</span><b>Chưa có kết quả</b><small>Chọn ảnh, nhập số gram rồi bấm “Ước tính calo”.</small></div>}
        {state.status === 'loading' && <div className="calorieEmpty"><span className="calorieSpinner" /><b>Đang nhận diện món ăn…</b></div>}
        {state.status === 'error' && <p className="error">{state.error}</p>}

        {state.status === 'ok' && !state.data.is_food && (
          <div className="warn"><b>AI chưa xác nhận đây là ảnh thức ăn.</b><br/>Không tính calo để tránh đưa ra con số sai.</div>
        )}

        {state.status === 'ok' && state.data.is_food && (
          <div className="calorieResult">
            <div className="foodTitle">
              <div>
                <span className="foodKicker">MÓN ĂN NHẬN DIỆN</span>
                <h3>{state.data.food.name}</h3>
                <p>Điểm phù hợp {(state.data.food.match_score * 100).toFixed(1)}% · cosine {state.data.food.similarity.toFixed(3)}</p>
              </div>
              <div className="calorieBig"><b>{state.data.estimated_kcal}</b><span>kcal</span></div>
            </div>

            {state.data.uncertain && <div className="calorieWarning">⚠️ AI chưa chắc chắn. Hãy xem Top dự đoán và xác nhận món đúng ở danh sách bên dưới.</div>}

            <div className="calorieStats">
              <div><span>Khẩu phần</span><b>{state.data.portion_grams} g</b></div>
              <div><span>Năng lượng / 100g</span><b>{state.data.kcal_per_100g} kcal</b></div>
              <div><span>Khoảng tham khảo</span><b>{state.data.estimated_range_kcal?.[0]}–{state.data.estimated_range_kcal?.[1]} kcal</b></div>
            </div>

            <h4>Top dự đoán</h4>
            {state.data.top_predictions.map((p) => (
              <div className="caloriePrediction" key={p.label}>
                <span>{p.name}</span>
                <div className="track"><div className="fill" style={{ width: `${p.match_score * 100}%` }} /></div>
                <b>{(p.match_score * 100).toFixed(1)}%</b>
              </div>
            ))}

            <div className="manualFoodBox">
              <label><b>AI nhận sai?</b> Chọn lại món gần đúng để tính lại calo:</label>
              <select value={manualLabel} onChange={(e) => recalculate(e.target.value)}>
                {(state.data.correction_options || []).map((f) => <option key={f.label} value={f.label}>{f.name} · {f.kcal_per_100g} kcal/100g</option>)}
              </select>
              {selectedOption && <small>Đang dùng bảng tham khảo: {selectedOption.name}</small>}
            </div>

            <div className="calorieWarning">⚠️ {state.data.warning}</div>
            <p className="muted">⏱ {state.data.latency_ms} ms</p>
          </div>
        )}
      </div>
    </section>
  );
}
