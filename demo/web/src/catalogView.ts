export type ModelStatus = "available" | "cached" | "downloading" | "ready";

export type CatalogModel = {
  key: string;
  model_id: string;
  display_name: string;
  role: "ner" | "pii";
  category: string;
  formats: string[];
  languages: string[];
  param_count: number | null;
  status: ModelStatus;
  error: string | null;
  progress: { files_done: number; files_total: number } | null;
};

const STATUS_RANK: Record<ModelStatus, number> = {
  downloading: 0,
  ready: 1,
  cached: 2,
  available: 3,
};

/** Prefer active, in-progress, and smaller English models in the picker. */
export function visibleModels(
  models: CatalogModel[],
  role: "ner" | "pii",
  query: string,
  activeKey: string,
): CatalogModel[] {
  const pool = models.filter((model) => model.role === role);
  const needle = query.trim().toLowerCase();
  const matched = needle
    ? pool.filter((model) =>
        [model.key, model.model_id, model.display_name, model.status]
          .join(" ")
          .toLowerCase()
          .includes(needle),
      )
    : [...pool].sort((left, right) => {
        const statusDelta = STATUS_RANK[left.status] - STATUS_RANK[right.status];
        if (statusDelta !== 0) {
          return statusDelta;
        }
        const leftEnglish = left.languages.includes("en") ? 0 : 1;
        const rightEnglish = right.languages.includes("en") ? 0 : 1;
        if (leftEnglish !== rightEnglish) {
          return leftEnglish - rightEnglish;
        }
        return (left.param_count ?? Number.MAX_SAFE_INTEGER) -
          (right.param_count ?? Number.MAX_SAFE_INTEGER);
      });

  const seen = new Set<string>();
  const limited: CatalogModel[] = [];
  const active = pool.find((model) => model.key === activeKey);
  if (active) {
    limited.push(active);
    seen.add(active.key);
  }
  for (const model of matched) {
    if (seen.has(model.key)) {
      continue;
    }
    if (!needle && model.model_id.includes("onnx") && model.status === "available") {
      continue;
    }
    limited.push(model);
    seen.add(model.key);
    if (limited.length >= 40) {
      break;
    }
  }
  return limited;
}
