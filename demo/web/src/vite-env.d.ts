/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OPENMED_DEMO_API_ORIGIN: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
