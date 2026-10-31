import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import svgrPlugin from 'vite-plugin-svgr';
import eslintPlugin from 'vite-plugin-eslint';

// https://vitejs.dev/config/
export default defineConfig({
  base: '/',
  build: {
    outDir: 'dist',
    // The MUI vendor chunk is ~540 kB (166 kB gzip) on its own; it can't be split further without
    // circular chunks, and app code is split per route, so warn only above that baseline.
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // Stable vendor libraries in their own chunks: smaller entry bundle, better caching
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          mui: ['@mui/material', '@mui/system', '@emotion/react', '@emotion/styled'],
          query: ['@tanstack/react-query', 'axios', 'zustand', 'notistack'],
        },
      },
    },
  },
  plugins: [
    react(),
    svgrPlugin({
      svgrOptions: {
        icon: true,
        // ...svgr options (https://react-svgr.com/docs/options/)
      },
    }),
    eslintPlugin({
      include: ['src/**/*.jsx', 'src/**/*.js', 'src/**/*.ts', 'src/**/*.tsx'],
      exclude: ['node_modules/**', 'dist/**', 'build/**', '**/*.mdx', '**/*.md'],
    }),
  ],
  test: {
    globals: true,
    environment: 'jsdom',
  },
});
