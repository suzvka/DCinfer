import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// playground 即 vite 应用根（index.html 在本目录）；src 以源码形式被直接消费
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
});
