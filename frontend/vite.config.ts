import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // SSE / long requests: avoid default proxy timeouts closing the stream in dev.
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
})
