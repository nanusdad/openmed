export type EntitySpan = {
  text: string;
  label: string;
  confidence: number;
  start: number;
  end: number;
};

export type HighlightPart =
  | { kind: "text"; text: string }
  | { kind: "span"; text: string; label: string; confidence: number };

/** Split source text into plain runs and non-overlapping entity spans. */
export function highlightParts(source: string, spans: EntitySpan[]): HighlightPart[] {
  const sorted = spans
    .filter(
      (span) =>
        Number.isFinite(span.start) &&
        Number.isFinite(span.end) &&
        span.start >= 0 &&
        span.end > span.start &&
        span.end <= source.length,
    )
    .sort((left, right) => left.start - right.start || right.end - left.end);
  const chosen: EntitySpan[] = [];
  let cursor = 0;
  for (const span of sorted) {
    if (span.start < cursor) {
      continue;
    }
    chosen.push(span);
    cursor = span.end;
  }

  const parts: HighlightPart[] = [];
  let index = 0;
  for (const span of chosen) {
    if (span.start > index) {
      parts.push({ kind: "text", text: source.slice(index, span.start) });
    }
    parts.push({
      kind: "span",
      text: source.slice(span.start, span.end),
      label: span.label,
      confidence: span.confidence,
    });
    index = span.end;
  }
  if (index < source.length) {
    parts.push({ kind: "text", text: source.slice(index) });
  }
  return parts;
}

export function labelHue(label: string): number {
  let hash = 0;
  for (const char of label) {
    hash = (hash * 33 + char.charCodeAt(0)) >>> 0;
  }
  return hash % 360;
}

export function formatConfidence(confidence: number): string {
  if (!Number.isFinite(confidence)) {
    return "—";
  }
  return `${Math.round(confidence * 100)}%`;
}
