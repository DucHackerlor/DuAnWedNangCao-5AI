import { useEffect, useState } from 'react';

// Chọn ảnh + xem trước. Giải phóng object URL khi đổi ảnh để tránh rò bộ nhớ.
export default function ImagePicker({ onChange, label = 'Chọn ảnh' }) {
  const [preview, setPreview] = useState(null);
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  return (
    <div className="picker">
      <label className="button">
        {label}
        <input type="file" accept="image/*" hidden onChange={(e) => {
          const file = e.target.files?.[0];
          if (!file) return;
          setPreview(URL.createObjectURL(file));
          onChange(file);
        }} />
      </label>
      {preview && <img src={preview} alt="Ảnh đầu vào" className="preview" />}
    </div>
  );
}