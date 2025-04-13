import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Useful if you're using chartjs-plugin-zoom in your code
      'chartjs-plugin-zoom': path.resolve(__dirname, 'node_modules/chartjs-plugin-zoom'),
    },
  },
  server: {
    proxy: {
      // 🚀 Dev-time proxy: redirects all `/api` calls to your FastAPI backend
      '/api': {
        target: 'http://localhost:8000', // FastAPI server
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
