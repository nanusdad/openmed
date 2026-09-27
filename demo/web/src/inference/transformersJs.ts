import type {
  AnalyzeResult,
  CatalogResponse,
  DeidentifyMethod,
  DeidentifyResult,
  ExtractResult,
  InferencePort,
} from "./port";

/**
 * Optional later adapter. The browser demo does not load Transformers.js,
 * download weights in the page, or send text anywhere.
 */
export class TransformersJsInference implements InferencePort {
  analyze(_text: string, _modelId?: string): Promise<AnalyzeResult> {
    return unsupported();
  }

  extractPii(_text: string, _modelId?: string): Promise<ExtractResult> {
    return unsupported();
  }

  deidentify(
    _text: string,
    _method: DeidentifyMethod,
    _modelId?: string,
  ): Promise<DeidentifyResult> {
    return unsupported();
  }

  listModels(): Promise<CatalogResponse> {
    return unsupported();
  }

  pullModel(_modelId: string): Promise<{ model_name: string; status: string }> {
    return unsupported();
  }

  selectModels(_selection: {
    nerModel?: string;
    piiModel?: string;
  }): Promise<CatalogResponse> {
    return unsupported();
  }
}

function unsupported(): Promise<never> {
  return Promise.reject(new Error("transformers_js_not_enabled"));
}
