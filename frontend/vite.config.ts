import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev only: the browser calls /api/*, Vite forwards to FastAPI on :8000 and strips
// the /api prefix. Same-origin from the browser's view, so no CORS config is needed
// yet. Deployment needs a real answer (single origin or an explicit CORS allow-list).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
