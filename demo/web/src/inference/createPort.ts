import { LocalHttpInference } from "./localHttp";
import type { InferencePort } from "./port";
import { TransformersJsInference } from "./transformersJs";

export type InferenceKind = "local-http" | "transformers-js";

export function readApiOrigin(): string {
  const origin = import.meta.env.VITE_OPENMED_DEMO_API_ORIGIN;
  if (!origin || !origin.trim()) {
    throw new Error("VITE_OPENMED_DEMO_API_ORIGIN is required");
  }
  return origin.trim().replace(/\/$/, "");
}

/** Build the UI inference adapter. Local HTTP is the supported path. */
export function createInferencePort(kind: InferenceKind = "local-http"): InferencePort {
  if (kind === "transformers-js") {
    return new TransformersJsInference();
  }
  return new LocalHttpInference(readApiOrigin());
}
