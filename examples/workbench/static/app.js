const LABEL_COLORS = {
  DISEASE: "#8e2f1f",
  CONDITION: "#8e2f1f",
  PATHOLOGY: "#8e2f1f",
  DRUG: "#1f5f86",
  CHEM: "#1f5f86",
  MEDICATION: "#1f5f86",
  CANCER: "#7a3e6a",
  CELL: "#7a3e6a",
  GENE_OR_GENE_PRODUCT: "#3d6b3a",
  GENE: "#3d6b3a",
  PROTEIN: "#3d6b3a",
  DNA: "#3d6b3a",
  ORGAN: "#8a5a24",
  TISSUE: "#8a5a24",
  ANATOMY: "#8a5a24",
  ORGANISM: "#2f5d50",
  SPECIES: "#2f5d50",
  FIRST_NAME: "#23483e",
  LAST_NAME: "#23483e",
  NAME: "#23483e",
  EMAIL: "#1f5f86",
  PHONE: "#3d6b3a",
  PHONE_NUMBER: "#3d6b3a",
  SSN: "#8e2f1f",
  DATE: "#7a3e6a",
  DATE_OF_BIRTH: "#7a3e6a",
  ADDRESS: "#8a5a24",
  STREET_ADDRESS: "#8a5a24",
  ID: "#5c4a8a",
  MEDICAL_RECORD_NUMBER: "#5c4a8a",
};

const state = {
  catalog: null,
  scenarioId: null,
  lastSample: "",
};

const scenarioList = document.querySelector("#scenario-list");
const modelSelect = document.querySelector("#model");
const modelNote = document.querySelector("#model-note");
const note = document.querySelector("#note");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#threshold-value");
const form = document.querySelector("#bench");
const runButton = document.querySelector("#run");
const statusEl = document.querySelector("#status");
const errorEl = document.querySelector("#error");
const marked = document.querySelector("#marked");
const redactedWrap = document.querySelector("#redacted-wrap");
const redacted = document.querySelector("#redacted");
const timing = document.querySelector("#timing");
const ledger = document.querySelector("#ledger");
const ledgerBody = ledger.querySelector("tbody");
const ledgerEmpty = document.querySelector("#ledger-empty");

function colorFor(label) {
  const key = String(label || "").toUpperCase().replace(/[\s-]+/g, "_");
  return LABEL_COLORS[key] || "#2f5d50";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function currentScenario() {
  return state.catalog.scenarios.find((item) => item.id === state.scenarioId);
}

function modelInfo(key) {
  return state.catalog.models[key];
}

function optionLabel(info) {
  const size = info.size_mb ? ` · ${info.size_mb} MB` : "";
  return `${info.display_name}${size}`;
}

function renderScenarios() {
  scenarioList.replaceChildren();
  for (const scenario of state.catalog.scenarios) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scenario";
    button.dataset.id = scenario.id;
    button.innerHTML = `<strong>${escapeHtml(scenario.name)}</strong><small>${escapeHtml(scenario.summary)}</small>`;
    if (scenario.id === state.scenarioId) {
      button.setAttribute("aria-current", "true");
    }
    button.addEventListener("click", () => selectScenario(scenario.id));
    scenarioList.append(button);
  }
}

function fillModels(scenario) {
  const previous = modelSelect.value;
  modelSelect.replaceChildren();

  const recommended = document.createElement("optgroup");
  recommended.label = "For this scenario";
  for (const key of scenario.models) {
    const info = modelInfo(key);
    if (!info) continue;
    const option = document.createElement("option");
    option.value = key;
    option.textContent = optionLabel(info);
    recommended.append(option);
  }
  modelSelect.append(recommended);

  const used = new Set(scenario.models);
  for (const [category, keys] of Object.entries(state.catalog.groups)) {
    const extras = keys.filter((key) => !used.has(key));
    if (!extras.length) continue;
    const group = document.createElement("optgroup");
    group.label = category;
    for (const key of extras) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = optionLabel(modelInfo(key));
      group.append(option);
    }
    modelSelect.append(group);
  }

  if (previous && modelInfo(previous) && scenario.models.includes(previous)) {
    modelSelect.value = previous;
  } else if (scenario.models[0]) {
    modelSelect.value = scenario.models[0];
  }
  syncModelNote();
}

function syncModelNote() {
  const info = modelInfo(modelSelect.value);
  if (!info) {
    modelNote.textContent = "";
    return;
  }
  const kinds = info.entity_types.slice(0, 6).join(", ");
  modelNote.textContent = `${info.description} Looks for ${kinds}.`;
  threshold.value = String(info.recommended_confidence);
  thresholdValue.textContent = Number(info.recommended_confidence).toFixed(2);
}

function selectScenario(id, { forceSample = false } = {}) {
  state.scenarioId = id;
  const scenario = currentScenario();
  const untouched = note.value.trim() === "" || note.value === state.lastSample;
  if (forceSample || untouched) {
    note.value = scenario.sample;
    state.lastSample = scenario.sample;
  }
  renderScenarios();
  fillModels(scenario);
  clearResult();
}

function taskFor(scenario, modelKey) {
  const info = modelInfo(modelKey);
  if (scenario.task === "deidentify" && info && info.category === "Privacy") {
    return "deidentify";
  }
  if (info && info.category === "Privacy") return "pii";
  return "analyze";
}

function clearResult() {
  errorEl.hidden = true;
  errorEl.textContent = "";
  statusEl.textContent = "";
  timing.textContent = "";
  marked.textContent = "Mark a note to see entities written back onto the text.";
  redactedWrap.hidden = true;
  redacted.textContent = "";
  ledger.hidden = true;
  ledgerBody.replaceChildren();
  ledgerEmpty.hidden = false;
}

function renderMarked(text, entities) {
  const usable = entities
    .filter((entity) => Number.isInteger(entity.start) && Number.isInteger(entity.end) && entity.end > entity.start)
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const taken = [];
  const spans = [];
  for (const entity of usable) {
    const overlaps = taken.some((span) => entity.start < span.end && entity.end > span.start);
    if (overlaps) continue;
    taken.push(entity);
    spans.push(entity);
  }

  if (!spans.length) {
    marked.textContent = text;
    return;
  }

  let cursor = 0;
  let html = "";
  for (const entity of spans) {
    const start = Math.max(0, Math.min(entity.start, text.length));
    const end = Math.max(start, Math.min(entity.end, text.length));
    html += escapeHtml(text.slice(cursor, start));
    const color = colorFor(entity.label);
    html += `<span class="mark" style="--mark-color:${color}">${escapeHtml(text.slice(start, end))}<sup>${escapeHtml(entity.label)}</sup></span>`;
    cursor = end;
  }
  html += escapeHtml(text.slice(cursor));
  marked.innerHTML = html;
}

function renderLedger(entities) {
  ledgerBody.replaceChildren();
  if (!entities.length) {
    ledger.hidden = true;
    ledgerEmpty.hidden = false;
    ledgerEmpty.textContent = "This model did not mark anything above the score cutoff.";
    return;
  }
  ledgerEmpty.hidden = true;
  ledger.hidden = false;
  for (const entity of entities) {
    const row = document.createElement("tr");
    const color = colorFor(entity.label);
    row.innerHTML = `
      <td><span class="swatch" style="--mark-color:${color}"></span>${escapeHtml(entity.label)}</td>
      <td>${escapeHtml(entity.text)}</td>
      <td class="score">${Number(entity.confidence).toFixed(2)}</td>
    `;
    ledgerBody.append(row);
  }
}

async function runAnalysis(event) {
  event.preventDefault();
  const scenario = currentScenario();
  const text = note.value.trim();
  if (!text) {
    errorEl.hidden = false;
    errorEl.textContent = "Write a note before marking it.";
    return;
  }

  runButton.disabled = true;
  runButton.textContent = "Marking…";
  errorEl.hidden = true;
  statusEl.textContent = "Running the model. The first time, OpenMed downloads it, which can take a few minutes.";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model_name: modelSelect.value,
        task: taskFor(scenario, modelSelect.value),
        confidence_threshold: Number(threshold.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail || payload.error?.message || "The model run failed.";
      throw new Error(typeof detail === "string" ? detail : "The model run failed.");
    }

    const entities = payload.entities || [];
    renderMarked(payload.text || text, entities);
    renderLedger(entities);
    if (payload.deidentified_text) {
      redactedWrap.hidden = false;
      redacted.textContent = payload.deidentified_text;
    } else {
      redactedWrap.hidden = true;
    }
    const seconds = Number(payload.processing_time);
    timing.textContent = Number.isFinite(seconds) && seconds > 0 ? `${Math.round(seconds * 1000)} ms` : "";
    statusEl.textContent = entities.length
      ? `Marked ${entities.length} ${entities.length === 1 ? "span" : "spans"}.`
      : "Finished. Nothing was above the score cutoff.";
  } catch (error) {
    errorEl.hidden = false;
    errorEl.textContent = error.message || "The model run failed.";
    statusEl.textContent = "";
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Mark note";
  }
}

threshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(threshold.value).toFixed(2);
});
modelSelect.addEventListener("change", syncModelNote);
form.addEventListener("submit", runAnalysis);

fetch("/api/catalog")
  .then((response) => {
    if (!response.ok) throw new Error("Could not load the model list.");
    return response.json();
  })
  .then((catalog) => {
    state.catalog = catalog;
    selectScenario(catalog.scenarios[0].id, { forceSample: true });
  })
  .catch((error) => {
    errorEl.hidden = false;
    errorEl.textContent = error.message;
  });
