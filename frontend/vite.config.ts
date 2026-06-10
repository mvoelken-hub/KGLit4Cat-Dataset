import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const apiProxyTarget = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000';

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        '/api/v1': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
