const ERROR_TEXT: Record<string, string> = {
  invalid_request: "The request was rejected.",
  empty_text: "Paste a note first.",
  text_too_long: "That note is too long for this demo.",
  offline_model_missing:
    "That model is not in the local cache. Download it before running offline.",
  unknown_model: "That model is not in the local catalog.",
  invalid_model: "That model id is not accepted.",
  mlx_artifact_unavailable: "MLX is selected, and this model has no MLX weights.",
  inference_failed: "Local inference failed.",
  model_load_failed: "The model could not be loaded from the local cache.",
  hf_extra_missing: "Install the Hugging Face extra before downloading models.",
  transformers_js_not_enabled: "In-browser Transformers.js is not enabled in this demo.",
  network: "The local API did not respond. Start it on the configured loopback address and refresh.",
};

export function errorText(code: string): string {
  return ERROR_TEXT[code] ?? "The local demo could not complete that step.";
}
