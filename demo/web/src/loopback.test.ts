import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { isLoopbackHost } from "./loopback.ts";

describe("isLoopbackHost", () => {
  it("accepts loopback and rejects public binds", () => {
    assert.equal(isLoopbackHost("127.0.0.1"), true);
    assert.equal(isLoopbackHost("localhost"), true);
    assert.equal(isLoopbackHost("::1"), true);
    assert.equal(isLoopbackHost("0.0.0.0"), false);
    assert.equal(isLoopbackHost("example.com"), false);
  });
});
