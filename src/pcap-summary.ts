import { spawnSync } from "node:child_process";
import { readFileSync, statSync } from "node:fs";
import { basename } from "node:path";

const MAX_READ = 64 * 1024;
const TOOL_TIMEOUT_MS = 4_000;

function which(bin: string): string | undefined {
  const r = spawnSync("which", [bin], { encoding: "utf8", timeout: 1500 });
  const p = r.stdout?.trim();
  return r.status === 0 && p ? p : undefined;
}

function clipToolOut(text: string, n = 2_400): string {
  const t = text.replace(/\u0000/g, "").trim();
  if (t.length <= n) return t;
  return `${t.slice(0, n)}\n…[truncated]`;
}

function parsePcapHeader(buf: Buffer): { endian: "le" | "be"; version: string; snaplen: number; network: number } | undefined {
  if (buf.length < 24) return undefined;
  const magic = buf.readUInt32LE(0);
  const le = magic === 0xa1b2c3d4 || magic === 0xa1b23c4d || magic === 0xd4c3b2a1 || magic === 0x4d3cb2a1;
  const beMagic = buf.readUInt32BE(0);
  if (beMagic === 0xa1b2c3d4 || beMagic === 0xa1b23c4d) {
    return {
      endian: "be",
      version: `${buf.readUInt16BE(4)}.${buf.readUInt16BE(6)}`,
      snaplen: buf.readUInt32BE(16),
      network: buf.readUInt32BE(20),
    };
  }
  if (le || magic === 0xd4c3b2a1 || magic === 0x4d3cb2a1) {
    return {
      endian: "le",
      version: `${buf.readUInt16LE(4)}.${buf.readUInt16LE(6)}`,
      snaplen: buf.readUInt32LE(16),
      network: buf.readUInt32LE(20),
    };
  }
  if (buf[0] === 0x0a && buf[1] === 0x0d && buf[2] === 0x0d && buf[3] === 0x0a) {
    return { endian: "le", version: "pcapng", snaplen: 0, network: 0 };
  }
  return undefined;
}

function countClassicPackets(buf: Buffer, le: boolean): number {
  let off = 24;
  let n = 0;
  while (off + 16 <= buf.length && n < 4000) {
    const incl = le ? buf.readUInt32LE(off + 8) : buf.readUInt32BE(off + 8);
    if (incl > 0x40000) break;
    off += 16 + incl;
    n++;
  }
  return n;
}

function runCapinfos(path: string): string | undefined {
  const bin = which("capinfos");
  if (!bin) return undefined;
  const r = spawnSync(bin, ["-Tm", path], {
    encoding: "utf8",
    timeout: TOOL_TIMEOUT_MS,
    maxBuffer: 16_000,
  });
  if (r.error || r.status !== 0) {
    const r2 = spawnSync(bin, [path], { encoding: "utf8", timeout: TOOL_TIMEOUT_MS, maxBuffer: 16_000 });
    if (r2.error || r2.status !== 0) return undefined;
    return clipToolOut(r2.stdout || r2.stderr || "");
  }
  return clipToolOut(r.stdout || "");
}

function runTsharkSummary(path: string): string | undefined {
  const bin = which("tshark");
  if (!bin) return undefined;
  const r = spawnSync(bin, ["-r", path, "-q", "-z", "io,phs", "-c", "80"], {
    encoding: "utf8",
    timeout: TOOL_TIMEOUT_MS,
    maxBuffer: 24_000,
  });
  if (r.error || r.status !== 0) return undefined;
  return clipToolOut(r.stdout || "");
}

/** Bounded header/stats only — never returns packet payloads. */
export async function summarizePcapFile(path: string): Promise<string> {
  let size = 0;
  try {
    size = statSync(path).size;
  } catch (error) {
    return `Could not stat capture: ${error instanceof Error ? error.message : String(error)}`;
  }
  const name = basename(path);
  const lines = [`PCAP summary for ${name} (${size} bytes). Payloads omitted.`];
  let buf: Buffer;
  try {
    const raw = readFileSync(path);
    buf = raw.subarray(0, Math.min(MAX_READ, raw.length));
  } catch (error) {
    return `Could not read capture header: ${error instanceof Error ? error.message : String(error)}`;
  }
  const hdr = parsePcapHeader(buf);
  if (!hdr) {
    lines.push("Header was not a classic pcap or pcapng magic.");
  } else if (hdr.version === "pcapng") {
    lines.push("Format: pcapng (section header).");
  } else {
    const counted = countClassicPackets(buf, hdr.endian === "le");
    lines.push(
      `Format: pcap v${hdr.version} ${hdr.endian} · snaplen ${hdr.snaplen} · linktype ${hdr.network} · packets in first ${buf.length} bytes: ${counted}${size > MAX_READ ? "+" : ""}.`,
    );
  }
  const capinfos = runCapinfos(path);
  if (capinfos) {
    lines.push("capinfos:", capinfos);
  }
  const tshark = runTsharkSummary(path);
  if (tshark) {
    lines.push("tshark protocol hierarchy (truncated):", tshark);
  } else if (!capinfos) {
    lines.push(
      "tshark/capinfos are not installed, so this is a bounded header summary only. Install tshark for deep decode.",
    );
  }
  return lines.join("\n");
}
