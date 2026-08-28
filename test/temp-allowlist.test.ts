import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { summarizePcapFile } from "../src/pcap-summary.js";
import { assertTempAnalyzeFile, TempAnalyzeAllowlist } from "../src/temp-allowlist.js";

function tinyPcap(dir: string, name = "sample.pcap"): string {
  const hdr = Buffer.alloc(24);
  hdr.writeUInt32LE(0xa1b2c3d4, 0);
  hdr.writeUInt16LE(2, 4);
  hdr.writeUInt16LE(4, 6);
  hdr.writeUInt32LE(65535, 16);
  hdr.writeUInt32LE(1, 20);
  const pktHdr = Buffer.alloc(16);
  pktHdr.writeUInt32LE(14, 8);
  pktHdr.writeUInt32LE(14, 12);
  const eth = Buffer.alloc(14);
  const file = join(dir, name);
  writeFileSync(file, Buffer.concat([hdr, pktHdr, eth]));
  return file;
}

test("temp analyze allowlist add/analyze/remove is idempotent and does not grant write", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pcap-"));
  const file = tinyPcap(dir);
  const allow = new TempAnalyzeAllowlist();
  const grant = allow.add(file);
  assert.equal(grant.path, file);
  assert.equal(allow.has(file), true);
  assert.equal(allow.list().length, 1);
  assert.equal(allow.remove(file), true);
  assert.equal(allow.has(file), false);
  assert.equal(allow.remove(file), false);
  assert.equal(allow.list().length, 0);
});

test("temp analyze rejects .., directories, and non-pcap files", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pcap-bad-"));
  const file = tinyPcap(dir);
  const nested = join(dir, "sub");
  mkdirSync(nested);
  assert.throws(() => assertTempAnalyzeFile(`${nested}/../secret.pcap`), /\.\./);
  assert.throws(() => assertTempAnalyzeFile(`${dir}/../nope.pcap`), /\.\./);
  assert.throws(() => assertTempAnalyzeFile(dir), /directory/i);
  const txt = join(dir, "notes.txt");
  writeFileSync(txt, "not a capture");
  assert.throws(() => assertTempAnalyzeFile(txt), /pcap/i);
  const junk = join(dir, "fake.pcap");
  writeFileSync(junk, "hello world this is not pcap");
  assert.throws(() => assertTempAnalyzeFile(junk), /magic/i);
  assert.ok(assertTempAnalyzeFile(file));
});

test("temp analyze TTL expiry drops the grant", async () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pcap-ttl-"));
  const file = tinyPcap(dir);
  const allow = new TempAnalyzeAllowlist();
  allow.add(file, 20);
  assert.equal(allow.has(file), true);
  await delay(40);
  assert.equal(allow.has(file), false);
  assert.equal(allow.list().length, 0);
});

test("pcap summary is bounded and does not include payloads", async () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pcap-sum-"));
  const file = tinyPcap(dir);
  const text = await summarizePcapFile(file);
  assert.match(text, /PCAP summary/);
  assert.doesNotMatch(text, /\n[0-9a-f]{32,}/i);
  assert.match(text, /payloads omitted/i);
});
