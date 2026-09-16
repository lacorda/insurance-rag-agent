import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // macOS 上 Vite 默认只听 [::1]，Flask 壳引用的是 127.0.0.1，脚本会加载失败导致空白页
    host: "127.0.0.1",
    // 与 Flask VITE_ORIGIN 对齐；5173 被占用时直接失败，禁止自动换端口
    port: 5173,
    strictPort: true,
    cors: true,
    proxy: {
      "/api": "http://127.0.0.1:5000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/index.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/index.[ext]",
      },
    },
  },
});
