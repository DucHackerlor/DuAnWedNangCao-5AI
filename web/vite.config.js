import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Khi dev (npm run dev), /api được chuyển sang FastAPI ở cổng 8000.
// Khi build, FastAPI phục vụ luôn thư mục dist → cùng origin, không cần CORS.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
});