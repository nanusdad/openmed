import type { CatalogModel } from "../catalogView";
import type { EntitySpan } from "../highlight";

export type CatalogResponse = {
  active: {
    ner_model: string;
    pii_model: string;
    resolved_ner_model: string;
    resolved_pii_model: string;
    backend: string;
    preferred_backend: string;
    offline: boolean;
  };
  platform: {
    system: string;
    machine: string;
    mlx_available: boolean;
  };
  models: CatalogModel[];
};

export type AnalyzeResult = {
  text: string;
  entities: EntitySpan[];
  model_name: string;
};

export type ExtractResult = {
  text: string;
  entities: EntitySpan[];
  model_name: string;
};

export type DeidentifyResult = {
  original_text: string;
  deidentified_text: string;
  pii_entities: EntitySpan[];
  method: "mask" | "replace";
  model_name: string;
};

export type DeidentifyMethod = "mask" | "replace";

/**
 * The only inference seam in the UI.
 * Local HTTP is the current adapter. Transformers.js can implement this later.
 */
export interface InferencePort {
  analyze(text: string, modelId?: string): Promise<AnalyzeResult>;
  extractPii(text: string, modelId?: string): Promise<ExtractResult>;
  deidentify(
    text: string,
    method: DeidentifyMethod,
    modelId?: string,
  ): Promise<DeidentifyResult>;
  listModels(): Promise<CatalogResponse>;
  pullModel(modelId: string): Promise<{ model_name: string; status: string }>;
  selectModels(selection: {
    nerModel?: string;
    piiModel?: string;
  }): Promise<CatalogResponse>;
}
