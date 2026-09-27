import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { localStatusLine } from "./statusLine.ts";

describe("localStatusLine", () => {
  it("uses the configured loopback host", () => {
    assert.equal(
      localStatusLine("http://127.0.0.1:8765"),
      "Local · 127.0.0.1 · no cloud PHI",
    );
  });
});
