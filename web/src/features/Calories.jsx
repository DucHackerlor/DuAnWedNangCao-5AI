import { useState } from 'react';
import { postImage } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Calories({ disabled = false }) {
  const [file, setFile] = useState(null);
  const [grams, setGrams] = useState(200);
  const [state, setState] = useState({ status: 'idle' });

  async function run() {
    if (!file || disabled || state.status === 'loading') return;
    setState({ status: 'loading' });

    try {
      const data = await postImage('/api/calorie', file, { grams, top_k: 3 });
      setState({ status: 'ok', data });
    } catch (err) {
      setState({ status: 'error', error: err.message });
    }
  }

  return (
    <section className="grid calorieGrid">
      <div>
        <h2>AI ước tính calo món ăn</h2>
        <p className="muted">
          AI Food-101 nhận diện món ăn từ ảnh, sau đó tính năng lượng theo khối lượng bạn nhập.
        </p>

        <ImagePicker
          label="Chọn ảnh món ăn"
          onChange={(f) => {
            setFile(f);
            setState({ status: 'idle' });
          }}
        />

        <div className="calorieControls">
          <label>
            Khối lượng khẩu phần
            <b>{grams} g</b>
          </label>
          <input
            type="range"
            min="20"
            max="1000"
            step="10"
            value={grams}
            onChange={(e) => setGrams(Number(e.target.value))}
          />
          <input
            className="gramsInput"
            type="number"
            min="1"
            max="3000"
            value={grams}
            onChange={(e) => setGrams(Math.max(1, Math.min(3000, Number(e.target.value) || 1)))}
          />
        </div>

        <button
          type="button"
          className="calorieButton"
          disabled={!file || disabled || state.status === 'loading'}
          onClick={run}
        >
          {state.status === 'loading' ? 'AI đang phân tích…' : 'Ước tính calo'}
        </button>

        <p className="calorieHint">
          Lần sử dụng đầu tiên có thể lâu hơn vì máy cần tải model Food-101.
        </p>
      </div>

      <div>
        {state.status === 'idle' && (
          <div className="calorieEmpty">
            <span>🍽️</span>
            <b>Chưa có kết quả</b>
            <small>Chọn ảnh món ăn, nhập số gram rồi bấm “Ước tính calo”.</small>
          </div>
        )}

        {state.status === 'loading' && (
          <div className="calorieEmpty">
            <span className="calorieSpinner" />
            <b>Đang nhận diện món ăn…</b>
            <small>Lần đầu có thể mất thêm thời gian để tải model.</small>
          </div>
        )}

        {state.status === 'error' && <p className="error">{state.error}</p>}

        {state.status === 'ok' && (
          <div className="calorieResult">
            <div className="foodTitle">
              <div>
                <span className="foodKicker">MÓN ĂN NHẬN DIỆN</span>
                <h3>{state.data.food.name}</h3>
                <p>Điểm phù hợp {(state.data.food.confidence * 100).toFixed(1)}%</p>
              </div>
              <div className="calorieBig">
                <b>{state.data.estimated_kcal}</b>
                <span>kcal</span>
              </div>
            </div>

            <div className="calorieStats">
              <div><span>Khẩu phần</span><b>{state.data.portion_grams} g</b></div>
              <div><span>Năng lượng / 100g</span><b>{state.data.kcal_per_100g} kcal</b></div>
              <div><span>Khoảng tham khảo</span><b>{state.data.estimated_range_kcal[0]}–{state.data.estimated_range_kcal[1]} kcal</b></div>
            </div>

            <h4>Top dự đoán</h4>
            {state.data.top_predictions.map((p) => (
              <div className="caloriePrediction" key={p.label}>
                <span>{p.name}</span>
                <div className="track">
                  <div className="fill" style={{ width: `${p.confidence * 100}%` }} />
                </div>
                <b>{(p.confidence * 100).toFixed(1)}%</b>
              </div>
            ))}

            {state.data.uncertain && (
              <div className="calorieWarning">
                ⚠️ AI chưa chắc chắn về món này. Hãy xem Top dự đoán và coi calo là tham khảo.
              </div>
            )}
            <div className="calorieWarning">
              ⚠️ {state.data.warning}
            </div>
            <p className="muted">⏱ {state.data.latency_ms} ms</p>
          </div>
        )}
      </div>
    </section>
  );
}
