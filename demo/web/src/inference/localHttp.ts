import type {
  AnalyzeResult,
  CatalogResponse,
  DeidentifyMethod,
  DeidentifyResult,
  ExtractResult,
  InferencePort,
} from "./port";

export class LocalHttpInference implements InferencePort {
  constructor(private readonly origin: string) {}

  analyze(text: string, modelId?: string): Promise<AnalyzeResult> {
    return this.post<AnalyzeResult>("/analyze", {
      text,
      ...(modelId ? { model_name: modelId } : {}),
    });
  }

  extractPii(text: string, modelId?: string): Promise<ExtractResult> {
    return this.post<ExtractResult>("/pii/extract", {
      text,
      ...(modelId ? { model_name: modelId } : {}),
    });
  }

  deidentify(
    text: string,
    method: DeidentifyMethod,
    modelId?: string,
  ): Promise<DeidentifyResult> {
    return this.post<DeidentifyResult>("/pii/deidentify", {
      text,
      method,
      ...(modelId ? { model_name: modelId } : {}),
    });
  }

  listModels(): Promise<CatalogResponse> {
    return this.send<CatalogResponse>("/models", { method: "GET" });
  }

  pullModel(modelId: string): Promise<{ model_name: string; status: string }> {
    return this.post("/models/pull", { model_name: modelId });
  }

  selectModels(selection: {
    nerModel?: string;
    piiModel?: string;
  }): Promise<CatalogResponse> {
    return this.post<CatalogResponse>("/models/active", {
      ...(selection.nerModel ? { ner_model: selection.nerModel } : {}),
      ...(selection.piiModel ? { pii_model: selection.piiModel } : {}),
    });
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return this.send<T>(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  private async send<T>(path: string, init: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(new URL(path, this.origin), {
        ...init,
        headers: { accept: "application/json", ...(init.headers ?? {}) },
        referrerPolicy: "no-referrer",
        credentials: "omit",
      });
    } catch {
      throw new Error("network");
    }
    const payload = (await response.json().catch(() => ({}))) as { error?: string };
    if (!response.ok) {
      throw new Error(payload.error || "inference_failed");
    }
    return payload as T;
  }
}
