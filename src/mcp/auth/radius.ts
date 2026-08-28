import { createHash, randomBytes } from "node:crypto";
import { createSocket, type Socket } from "node:dgram";
import type { PasswordAuthenticator, PasswordLogin } from "./types.js";

const ACCESS_REQUEST = 1;
const ACCESS_ACCEPT = 2;
const ACCESS_REJECT = 3;
const ATTR_USER_NAME = 1;
const ATTR_USER_PASSWORD = 2;
const ATTR_NAS_IP = 4;
const ATTR_NAS_IDENTIFIER = 32;
const ATTR_FILTER_ID = 11;

export type RadiusSend = (packet: Buffer, host: string, port: number) => Promise<Buffer>;

export type RadiusOptions = {
  host: string;
  port: number;
  secret: string;
  nasIdentifier: string;
  allowedFilterIds: string[];
  send?: RadiusSend;
  timeoutMs?: number;
};

function md5(data: Buffer): Buffer {
  return createHash("md5").update(data).digest();
}

/** RFC 2865 User-Password hiding. */
export function hideRadiusPassword(password: string, secret: string, authenticator: Buffer): Buffer {
  const padded = Buffer.concat([Buffer.from(password, "utf8")]);
  const extra = (16 - (padded.length % 16)) % 16;
  const block = extra ? Buffer.concat([padded, Buffer.alloc(extra)]) : padded;
  const out = Buffer.alloc(block.length);
  let last = authenticator;
  for (let i = 0; i < block.length; i += 16) {
    const hash = md5(Buffer.concat([Buffer.from(secret, "utf8"), last]));
    for (let j = 0; j < 16; j++) {
      out[i + j] = (block[i + j] ?? 0) ^ (hash[j] ?? 0);
    }
    last = out.subarray(i, i + 16);
  }
  return out;
}

function attr(type: number, value: Buffer): Buffer {
  const buf = Buffer.alloc(2 + value.length);
  buf[0] = type;
  buf[1] = buf.length;
  value.copy(buf, 2);
  return buf;
}

export function encodeAccessRequest(opts: {
  identifier: number;
  authenticator: Buffer;
  username: string;
  password: string;
  secret: string;
  nasIdentifier: string;
}): Buffer {
  const nasIp = attr(ATTR_NAS_IP, Buffer.from([127, 0, 0, 1]));
  const parts = [
    attr(ATTR_USER_NAME, Buffer.from(opts.username, "utf8")),
    attr(ATTR_USER_PASSWORD, hideRadiusPassword(opts.password, opts.secret, opts.authenticator)),
    nasIp,
    attr(ATTR_NAS_IDENTIFIER, Buffer.from(opts.nasIdentifier, "utf8")),
  ];
  const attrs = Buffer.concat(parts);
  const length = 20 + attrs.length;
  const packet = Buffer.alloc(length);
  packet[0] = ACCESS_REQUEST;
  packet[1] = opts.identifier & 0xff;
  packet.writeUInt16BE(length, 2);
  opts.authenticator.copy(packet, 4);
  attrs.copy(packet, 20);
  return packet;
}

export function encodeAccessResponse(opts: {
  code: number;
  identifier: number;
  filterIds?: string[];
}): Buffer {
  const parts = (opts.filterIds ?? []).map((id) => attr(ATTR_FILTER_ID, Buffer.from(id, "utf8")));
  const attrs = Buffer.concat(parts);
  const length = 20 + attrs.length;
  const packet = Buffer.alloc(length);
  packet[0] = opts.code;
  packet[1] = opts.identifier & 0xff;
  packet.writeUInt16BE(length, 2);
  randomBytes(16).copy(packet, 4);
  attrs.copy(packet, 20);
  return packet;
}

export const RADIUS_ACCESS_ACCEPT = ACCESS_ACCEPT;
export const RADIUS_ACCESS_REJECT = ACCESS_REJECT;

export function parseRadiusAttributes(packet: Buffer): { code: number; attrs: Map<number, string[]> } {
  const code = packet[0] ?? 0;
  const attrs = new Map<number, string[]>();
  let i = 20;
  while (i + 2 <= packet.length) {
    const type = packet[i] ?? 0;
    const len = packet[i + 1] ?? 0;
    if (len < 2 || i + len > packet.length) break;
    const value = packet.subarray(i + 2, i + len).toString("utf8");
    const list = attrs.get(type) ?? [];
    list.push(value);
    attrs.set(type, list);
    i += len;
  }
  return { code, attrs };
}

function udpSend(packet: Buffer, host: string, port: number, timeoutMs: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const socket: Socket = createSocket("udp4");
    const timer = setTimeout(() => {
      socket.close();
      reject(new Error("RADIUS timed out"));
    }, timeoutMs);
    socket.once("error", (err) => {
      clearTimeout(timer);
      socket.close();
      reject(err);
    });
    socket.once("message", (msg) => {
      clearTimeout(timer);
      socket.close();
      resolve(Buffer.from(msg));
    });
    socket.send(packet, port, host, (err) => {
      if (err) {
        clearTimeout(timer);
        socket.close();
        reject(err);
      }
    });
  });
}

function filterIdAllowed(got: string[], allow: string[]): boolean {
  if (!allow.length) return true;
  const want = new Set(allow.map((s) => s.trim().toLowerCase()).filter(Boolean));
  return got.some((g) => want.has(g.trim().toLowerCase()));
}

export function createRadiusAuthenticator(opts: RadiusOptions): PasswordAuthenticator {
  const timeoutMs = opts.timeoutMs ?? 3000;
  const send = opts.send ?? ((packet, host, port) => udpSend(packet, host, port, timeoutMs));
  return {
    method: "radius",
    async login(creds: PasswordLogin) {
      if (!opts.host.trim() || !opts.secret.trim()) {
        return { ok: false, reason: "RADIUS is not configured" };
      }
      if (!creds.username.trim() || !creds.password) {
        return { ok: false, reason: "username and password required" };
      }
      const authenticator = randomBytes(16);
      const identifier = randomBytes(1)[0] ?? 1;
      const packet = encodeAccessRequest({
        identifier,
        authenticator,
        username: creds.username.trim(),
        password: creds.password,
        secret: opts.secret,
        nasIdentifier: opts.nasIdentifier || "agent-orchestrator",
      });
      let response: Buffer;
      try {
        response = await send(packet, opts.host, opts.port);
      } catch {
        return { ok: false, reason: "RADIUS is not reachable" };
      }
      const parsed = parseRadiusAttributes(response);
      if (parsed.code === ACCESS_REJECT) return { ok: false, reason: "RADIUS rejected the user" };
      if (parsed.code !== ACCESS_ACCEPT) return { ok: false, reason: "RADIUS returned an unexpected code" };
      const filterIds = parsed.attrs.get(ATTR_FILTER_ID) ?? [];
      if (!filterIdAllowed(filterIds, opts.allowedFilterIds)) {
        return { ok: false, reason: "RADIUS Filter-Id is not in the allowlist" };
      }
      return { ok: true, subject: creds.username.trim(), groups: filterIds };
    },
  };
}
