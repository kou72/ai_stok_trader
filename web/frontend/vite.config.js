import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/start_training': 'http://localhost:5000',
      '/status': 'http://localhost:5000',
      '/logs': 'http://localhost:5000',
      '/csv_dirs': 'http://localhost:5000',
      '/api': 'http://localhost:5000',
    }
  }
})
