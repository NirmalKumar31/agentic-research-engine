import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // Served by FastAPI from the same origin in production, so no base path.
    outDir: "dist",
    sourcemap: false,
  },
  server: {
    // Dev only: the API runs separately during development, so same-origin
    // fetches are proxied rather than requiring CORS to be opened up.
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
