import assert from "node:assert/strict";
import { test } from "node:test";
import { HttpProvider } from "../src/providers/http.js";

test("http provider pins fetch to the configured host (no redirects)", async () => {
  const seen: Array<{ url: string; redirect?: RequestRedirect }> = [];
  const orig = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    seen.push({ url, redirect: init?.redirect });
    return new Response(JSON.stringify({ text: "ok" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const provider = new HttpProvider("http-lab", {
      type: "http",
      url: "http://127.0.0.1:9/agent",
    });
    const result = await provider.run({ prompt: "hi" });
    assert.equal(result.status, "finished");
    assert.equal(seen.length, 1);
    assert.equal(seen[0]?.url, "http://127.0.0.1:9/agent");
    assert.equal(seen[0]?.redirect, "error");
  } finally {
    globalThis.fetch = orig;
  }
});
