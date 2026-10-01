import { useEffect, useState } from 'react';

export default function ImagePicker({ onChange, label = 'Chọn ảnh', disabled = false }) {
  const [preview, setPreview] = useState(null);
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function pick(file) {
    if (!file || disabled) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    onChange(file);
  }

  return (
    <div className={`picker ${disabled ? 'disabled' : ''}`}>
      <label className="button">
        {label}
        <input type="file" accept="image/jpeg,image/png,image/webp" hidden disabled={disabled}
               onChange={(e) => pick(e.target.files?.[0])} />
      </label>
      {preview && <img src={preview} alt="Ảnh đầu vào" className="preview" />}
    </div>
  );
}
