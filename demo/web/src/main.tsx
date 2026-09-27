import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { createInferencePort, readApiOrigin } from "./inference/createPort";
import "./styles.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing root");
}

const origin = readApiOrigin();
createRoot(root).render(
  <StrictMode>
    <App port={createInferencePort("local-http")} apiOrigin={origin} />
  </StrictMode>,
);
