import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

import { isLoopbackHost } from "./src/loopback";

const demoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, demoRoot, "");
  const host = env.OPENMED_DEMO_WEB_HOST ?? "";
  const port = Number(env.OPENMED_DEMO_WEB_PORT);
  if (!isLoopbackHost(host)) {
    throw new Error("OPENMED_DEMO_WEB_HOST must be a loopback address");
  }
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("OPENMED_DEMO_WEB_PORT must be a port number");
  }
  if (!env.VITE_OPENMED_DEMO_API_ORIGIN) {
    throw new Error("VITE_OPENMED_DEMO_API_ORIGIN is required");
  }
  return {
    envDir: demoRoot,
    plugins: [react()],
    server: { host, port, strictPort: true },
    preview: { host, port, strictPort: true },
  };
});
