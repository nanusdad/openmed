import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { visibleModels, type CatalogModel } from "./catalogView.ts";

function model(overrides: Partial<CatalogModel> & Pick<CatalogModel, "key">): CatalogModel {
  return {
    model_id: overrides.key,
    display_name: overrides.key,
    role: "ner",
    category: "Disease",
    formats: ["pytorch"],
    languages: ["en"],
    param_count: 100,
    status: "available",
    error: null,
    progress: null,
    ...overrides,
  };
}

describe("visibleModels", () => {
  it("keeps the active model and filters by query", () => {
    const models = [
      model({ key: "disease-small", display_name: "Disease small", param_count: 10 }),
      model({ key: "pharma-large", display_name: "Pharma", param_count: 90, status: "ready" }),
      model({ key: "pii-small", role: "pii", display_name: "PII" }),
    ];
    const visible = visibleModels(models, "ner", "disease", "pharma-large");
    assert.deepEqual(
      visible.map((item) => item.key),
      ["pharma-large", "disease-small"],
    );
  });
});
