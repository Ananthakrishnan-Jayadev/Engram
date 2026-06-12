import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev proxy: the dashboard calls /api/* and Vite forwards to the FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
