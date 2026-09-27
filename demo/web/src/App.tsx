import { useEffect, useMemo, useState, type CSSProperties } from "react";

import { visibleModels, type CatalogModel } from "./catalogView";
import { errorText } from "./errors";
import { formatConfidence, highlightParts, labelHue, type EntitySpan } from "./highlight";
import type { CatalogResponse, DeidentifyMethod, InferencePort } from "./inference/port";
import { SAMPLES } from "./samples";
import { localStatusLine } from "./statusLine";

type RunMode = "ner" | "deidentify";

type ViewState = {
  mode: RunMode;
  source: string;
  output: string;
  entities: EntitySpan[];
  modelName: string;
  method: DeidentifyMethod;
};

export function App({ port, apiOrigin }: { port: InferencePort; apiOrigin: string }) {
  const statusLine = useMemo(() => localStatusLine(apiOrigin), [apiOrigin]);
  const [note, setNote] = useState(SAMPLES[0]?.text ?? "");
  const [method, setMethod] = useState<DeidentifyMethod>("mask");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState<ViewState | null>(null);
  const [shown, setShown] = useState(0);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [nerQuery, setNerQuery] = useState("");
  const [piiQuery, setPiiQuery] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const load = async () => {
      try {
        const next = await port.listModels();
        if (cancelled) {
          return;
        }
        setCatalog(next);
        if (next.models.some((model) => model.status === "downloading")) {
          timer = window.setTimeout(() => void load(), 800);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(errorText(caught instanceof Error ? caught.message : "network"));
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [port, refreshKey]);

  useEffect(() => {
    if (!view) {
      setShown(0);
      return;
    }
    let frame = 0;
    let count = 0;
    const step = () => {
      count = Math.min(view.entities.length, count + 6);
      setShown(count);
      if (count < view.entities.length) {
        frame = window.requestAnimationFrame(step);
      }
    };
    frame = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(frame);
  }, [view]);

  async function run(mode: RunMode) {
    setBusy(true);
    setError("");
    setView(null);
    try {
      if (mode === "ner") {
        const result = await port.analyze(note);
        setView({
          mode,
          source: result.text,
          output: "",
          entities: result.entities ?? [],
          modelName: result.model_name,
          method,
        });
      } else {
        const result = await port.deidentify(note, method);
        setView({
          mode,
          source: result.original_text,
          output: result.deidentified_text,
          entities: result.pii_entities ?? [],
          modelName: result.model_name,
          method: result.method,
        });
      }
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      setError(errorText(caught instanceof Error ? caught.message : "inference_failed"));
    } finally {
      setBusy(false);
    }
  }

  async function chooseModel(role: "ner" | "pii", key: string) {
    setError("");
    try {
      const next = await port.selectModels(
        role === "ner" ? { nerModel: key } : { piiModel: key },
      );
      setCatalog(next);
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      setError(errorText(caught instanceof Error ? caught.message : "unknown_model"));
    }
  }

  async function download(key: string) {
    setError("");
    try {
      await port.pullModel(key);
      setRefreshKey((value) => value + 1);
    } catch (caught) {
      setError(errorText(caught instanceof Error ? caught.message : "model_load_failed"));
    }
  }

  const visibleEntities = view?.entities.slice(0, shown) ?? [];
  const parts = view ? highlightParts(view.source, visibleEntities) : [];

  return (
    <div className="page">
      <header className="mast">
        <div>
          <p className="eyebrow">OpenMed</p>
          <h1>Local NER and de-identification</h1>
        </div>
        <p className="status" role="status">
          {statusLine}
        </p>
      </header>

      <p className="disclaimer">
        This is a local-first demo of on-device NER and de-identification. It is
        not a HIPAA compliance claim, and pasted text stays on this machine.
      </p>

      <main className="layout">
        <section className="panel composer" aria-label="Note">
          <div className="samples">
            {SAMPLES.map((sample) => (
              <button
                key={sample.id}
                type="button"
                className="chip"
                onClick={() => setNote(sample.text)}
              >
                {sample.title}
              </button>
            ))}
          </div>
          <label className="field-label" htmlFor="note">
            Synthetic note
          </label>
          <textarea
            id="note"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            spellCheck={false}
            placeholder="Paste a synthetic clinical note"
          />
          <p className="hint">
            Use the sample notes or your own fake text. Do not paste real
            patient information.
          </p>
          <div className="actions">
            <button type="button" disabled={busy || !note.trim()} onClick={() => void run("ner")}>
              {busy ? "Running locally…" : "Run NER"}
            </button>
            <label className="method">
              Method
              <select
                value={method}
                onChange={(event) => setMethod(event.target.value as DeidentifyMethod)}
              >
                <option value="mask">mask</option>
                <option value="replace">replace</option>
              </select>
            </label>
            <button
              type="button"
              className="secondary"
              disabled={busy || !note.trim()}
              onClick={() => void run("deidentify")}
            >
              De-identify
            </button>
          </div>
          {error ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}
        </section>

        <section className="panel result" aria-live="polite">
          <div className="result-head">
            <h2>{view?.mode === "deidentify" ? "De-identified" : "Entities"}</h2>
            {view ? <p className="meta">Model {view.modelName}</p> : null}
          </div>
          {!view && !busy ? (
            <p className="empty">Run NER or de-identification on the note.</p>
          ) : null}
          {busy ? <p className="empty">Running on this machine…</p> : null}
          {view ? (
            <>
              <div className="passage">
                {parts.map((part, index) =>
                  part.kind === "text" ? (
                    <span key={index}>{part.text}</span>
                  ) : (
                    <mark
                      key={index}
                      style={{ "--hue": String(labelHue(part.label)) } as CSSProperties}
                      title={`${part.label} ${formatConfidence(part.confidence)}`}
                    >
                      {part.text}
                      <span className="tag">
                        {part.label} {formatConfidence(part.confidence)}
                      </span>
                    </mark>
                  ),
                )}
              </div>
              {view.output ? (
                <pre className="redacted" aria-label="De-identified text">
                  {view.output}
                </pre>
              ) : null}
              <EntityTable entities={visibleEntities} />
            </>
          ) : null}
        </section>
      </main>

      <ModelPanel
        catalog={catalog}
        nerQuery={nerQuery}
        piiQuery={piiQuery}
        onNerQuery={setNerQuery}
        onPiiQuery={setPiiQuery}
        onSelect={(role, key) => void chooseModel(role, key)}
        onDownload={(key) => void download(key)}
      />
    </div>
  );
}

function EntityTable({ entities }: { entities: EntitySpan[] }) {
  if (entities.length === 0) {
    return <p className="empty">No spans yet.</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Label</th>
          <th>Span</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {entities.map((entity, index) => (
          <tr key={`${entity.start}-${entity.end}-${index}`}>
            <td>{entity.label}</td>
            <td>{entity.text}</td>
            <td>{formatConfidence(entity.confidence)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ModelPanel({
  catalog,
  nerQuery,
  piiQuery,
  onNerQuery,
  onPiiQuery,
  onSelect,
  onDownload,
}: {
  catalog: CatalogResponse | null;
  nerQuery: string;
  piiQuery: string;
  onNerQuery: (value: string) => void;
  onPiiQuery: (value: string) => void;
  onSelect: (role: "ner" | "pii", key: string) => void;
  onDownload: (key: string) => void;
}) {
  const active = catalog?.active;
  return (
    <section className="models" aria-label="Models">
      <div className="models-head">
        <h2>Models</h2>
        <p>
          {active
            ? `Backend ${active.preferred_backend} · ${
                active.offline ? "Hub offline after cache" : "Hub download allowed"
              }`
            : "Loading the local catalog…"}
        </p>
      </div>
      <div className="model-grid">
        <ModelPicker
          title="NER"
          role="ner"
          query={nerQuery}
          onQuery={onNerQuery}
          activeKey={active?.ner_model ?? ""}
          resolvedKey={active?.resolved_ner_model ?? ""}
          models={catalog?.models ?? []}
          onSelect={onSelect}
          onDownload={onDownload}
        />
        <ModelPicker
          title="PII / de-id"
          role="pii"
          query={piiQuery}
          onQuery={onPiiQuery}
          activeKey={active?.pii_model ?? ""}
          resolvedKey={active?.resolved_pii_model ?? ""}
          models={catalog?.models ?? []}
          onSelect={onSelect}
          onDownload={onDownload}
        />
      </div>
    </section>
  );
}

function ModelPicker({
  title,
  role,
  query,
  onQuery,
  activeKey,
  resolvedKey,
  models,
  onSelect,
  onDownload,
}: {
  title: string;
  role: "ner" | "pii";
  query: string;
  onQuery: (value: string) => void;
  activeKey: string;
  resolvedKey: string;
  models: CatalogModel[];
  onSelect: (role: "ner" | "pii", key: string) => void;
  onDownload: (key: string) => void;
}) {
  const choices = visibleModels(models, role, query, activeKey);
  const active = models.find((model) => model.key === activeKey);
  return (
    <div className="picker">
      <h3>{title}</h3>
      <label>
        Search
        <input value={query} onChange={(event) => onQuery(event.target.value)} />
      </label>
      <label>
        Active model
        <select value={activeKey} onChange={(event) => onSelect(role, event.target.value)}>
          {choices.map((model) => (
            <option key={model.key} value={model.key}>
              {model.display_name} · {model.status}
            </option>
          ))}
        </select>
      </label>
      <p className="meta">
        Status {active?.status ?? "available"}
        {resolvedKey && resolvedKey !== activeKey ? ` · weights ${resolvedKey}` : ""}
        {active?.progress && active.status === "downloading"
          ? ` · ${active.progress.files_done}/${active.progress.files_total || "?"}`
          : ""}
      </p>
      <button type="button" onClick={() => onDownload(activeKey)} disabled={!activeKey}>
        Download to local cache
      </button>
      {active?.error ? <p className="error">{errorText(active.error)}</p> : null}
    </div>
  );
}
