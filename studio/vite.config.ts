import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built app is served by the Model Finder web server at /studio/.
export default defineConfig({
  base: '/studio/',
  plugins: [react()],
  build: {
    outDir: '../src/model_finder/static/studio',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8765',
    },
  },
})
