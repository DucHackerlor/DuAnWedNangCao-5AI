export const API_BASE = import.meta.env.VITE_API_URL ?? '';
const DEFAULT_TIMEOUT = 180000;

async function parse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch {}
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function request(url, options = {}, timeout = DEFAULT_TIMEOUT) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, { ...options, signal: options.signal ?? controller.signal });
    return await parse(res);
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Yêu cầu quá lâu. Hãy thử lại.');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export function postImage(path, file, fields = {}) {
  const form = new FormData();
  form.append('file', file);
  Object.entries(fields).forEach(([k, v]) => form.append(k, String(v)));
  return request(`${API_BASE}${path}`, { method: 'POST', body: form });
}

export function postJson(path, body) {
  return request(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export const getHealth = () => request(`${API_BASE}/api/health`, {}, 7000);

export async function streamChat({ message, history, onSources, onToken, signal }) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  if (!res.body) throw new Error('Trình duyệt không hỗ trợ streaming.');
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';
    for (const ev of events) {
      if (!ev.startsWith('data: ')) continue;
      const data = JSON.parse(ev.slice(6));
      if (data.type === 'sources') onSources?.(data.items);
      if (data.type === 'token') onToken?.(data.text);
    }
  }
}
