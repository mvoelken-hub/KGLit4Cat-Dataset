import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, path.resolve(__dirname, '..'), '');
  const allowedHostsRaw = env.VITE_ALLOWED_HOSTS || '';
  const allowedHosts = allowedHostsRaw.split(',').map(h => h.trim()).filter(Boolean);

  return {
    plugins: [react()],
    server: {
      port: 3000,
      allowedHosts: allowedHosts.length > 0 ? allowedHosts : undefined,
      proxy: {
        '/api/v1': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
