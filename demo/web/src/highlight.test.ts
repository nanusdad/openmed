import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { formatConfidence, highlightParts } from "./highlight.ts";

describe("highlightParts", () => {
  it("keeps offsets and skips overlaps", () => {
    const parts = highlightParts("Casey Example has asthma.", [
      { text: "asthma", label: "DISEASE", confidence: 0.91, start: 18, end: 24 },
      { text: "Casey Example", label: "NAME", confidence: 0.5, start: 0, end: 13 },
      { text: "Example", label: "NAME", confidence: 0.2, start: 6, end: 13 },
    ]);
    assert.deepEqual(
      parts.map((part) => part.text),
      ["Casey Example", " has ", "asthma", "."],
    );
    const span = parts[2];
    assert.equal(span.kind, "span");
    if (span.kind === "span") {
      assert.equal(span.label, "DISEASE");
      assert.equal(span.confidence, 0.91);
    }
  });
});

describe("formatConfidence", () => {
  it("renders a percent", () => {
    assert.equal(formatConfidence(0.876), "88%");
  });
});
