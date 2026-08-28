import { connect as netConnect, type Socket } from "node:net";
import { connect as tlsConnect } from "node:tls";
import type { PasswordAuthenticator, PasswordLogin } from "./types.js";

export type LdapConnect = (opts: { host: string; port: number; tls: boolean }) => Promise<Socket>;

export type LdapOptions = {
  url: string;
  bindDnTemplate: string;
  serviceDn?: string;
  servicePassword?: string;
  baseDn: string;
  filter: string;
  groupAttr: string;
  allowedGroups: string[];
  tlsRejectUnauthorized: boolean;
  connect?: LdapConnect;
};

function encodeLength(len: number): Buffer {
  if (len < 128) return Buffer.from([len]);
  const bytes: number[] = [];
  let n = len;
  while (n > 0) {
    bytes.unshift(n & 0xff);
    n >>= 8;
  }
  return Buffer.from([0x80 | bytes.length, ...bytes]);
}

function tlv(tag: number, value: Buffer): Buffer {
  return Buffer.concat([Buffer.from([tag]), encodeLength(value.length), value]);
}

function encodeInt(n: number): Buffer {
  if (n === 0) return tlv(0x02, Buffer.from([0]));
  const bytes: number[] = [];
  let v = n;
  while (v > 0) {
    bytes.unshift(v & 0xff);
    v >>= 8;
  }
  if ((bytes[0] ?? 0) & 0x80) bytes.unshift(0);
  return tlv(0x02, Buffer.from(bytes));
}

function encodeOctet(text: string): Buffer {
  return tlv(0x04, Buffer.from(text, "utf8"));
}

function encodeSeq(tag: number, parts: Buffer[]): Buffer {
  return tlv(tag, Buffer.concat(parts));
}

function encodeBool(v: boolean): Buffer {
  return tlv(0x01, Buffer.from([v ? 0xff : 0x00]));
}

function encodeEnum(n: number): Buffer {
  return tlv(0x0a, Buffer.from([n & 0xff]));
}

export function encodeLdapBind(messageId: number, dn: string, password: string): Buffer {
  const bind = encodeSeq(0x60, [encodeInt(3), encodeOctet(dn), tlv(0x80, Buffer.from(password, "utf8"))]);
  return encodeSeq(0x30, [encodeInt(messageId), bind]);
}

export function encodeLdapSearch(opts: {
  messageId: number;
  baseDn: string;
  filterAttr: string;
  filterValue: string;
  attributes: string[];
}): Buffer {
  const eq = encodeSeq(0xa3, [encodeOctet(opts.filterAttr), encodeOctet(opts.filterValue)]);
  const attrs = encodeSeq(
    0x30,
    opts.attributes.map((a) => encodeOctet(a)),
  );
  const search = encodeSeq(0x63, [
    encodeOctet(opts.baseDn),
    encodeEnum(2),
    encodeEnum(0),
    encodeInt(0),
    encodeInt(0),
    encodeBool(false),
    eq,
    attrs,
  ]);
  return encodeSeq(0x30, [encodeInt(opts.messageId), search]);
}

export function encodeLdapUnbind(messageId: number): Buffer {
  return encodeSeq(0x30, [encodeInt(messageId), tlv(0x42, Buffer.alloc(0))]);
}

export function encodeLdapBindResponse(messageId: number, resultCode: number): Buffer {
  const op = encodeSeq(0x61, [encodeEnum(resultCode), encodeOctet(""), encodeOctet("")]);
  return encodeSeq(0x30, [encodeInt(messageId), op]);
}

export function encodeLdapSearchEntry(messageId: number, dn: string, groups: string[]): Buffer {
  const vals = encodeSeq(
    0x31,
    groups.map((g) => encodeOctet(g)),
  );
  const attr = encodeSeq(0x30, [encodeOctet("memberOf"), vals]);
  const attrs = encodeSeq(0x30, [attr]);
  const entry = encodeSeq(0x64, [encodeOctet(dn), attrs]);
  return encodeSeq(0x30, [encodeInt(messageId), entry]);
}

export function encodeLdapSearchDone(messageId: number, resultCode = 0): Buffer {
  const op = encodeSeq(0x65, [encodeEnum(resultCode), encodeOctet(""), encodeOctet("")]);
  return encodeSeq(0x30, [encodeInt(messageId), op]);
}

function readLength(buf: Buffer, offset: number): { len: number; size: number } {
  const first = buf[offset] ?? 0;
  if (first < 128) return { len: first, size: 1 };
  const count = first & 0x7f;
  let len = 0;
  for (let i = 0; i < count; i++) {
    len = (len << 8) | (buf[offset + 1 + i] ?? 0);
  }
  return { len, size: 1 + count };
}

type BerNode = { tag: number; value: Buffer; children: BerNode[] };

function parseBer(buf: Buffer, offset = 0): { node: BerNode; end: number } | undefined {
  if (offset >= buf.length) return undefined;
  const tag = buf[offset] ?? 0;
  const { len, size } = readLength(buf, offset + 1);
  const start = offset + 1 + size;
  const end = start + len;
  if (end > buf.length) return undefined;
  const value = buf.subarray(start, end);
  const constructed = Boolean(tag & 0x20);
  const children: BerNode[] = [];
  if (constructed) {
    let i = 0;
    while (i < value.length) {
      const child = parseBer(value, i);
      if (!child) break;
      children.push(child.node);
      i = child.end;
    }
  }
  return { node: { tag, value, children }, end };
}

function parseAll(buf: Buffer): BerNode[] {
  const out: BerNode[] = [];
  let i = 0;
  while (i < buf.length) {
    const parsed = parseBer(buf, i);
    if (!parsed) break;
    out.push(parsed.node);
    i = parsed.end;
  }
  return out;
}

function intValue(node: BerNode | undefined): number {
  if (!node || node.value.length === 0) return 0;
  let n = 0;
  for (const b of node.value) n = (n << 8) | b;
  return n;
}

function textValue(node: BerNode | undefined): string {
  return node ? node.value.toString("utf8") : "";
}

export function parseLdapMessages(buf: Buffer): Array<{ id: number; tag: number; resultCode?: number; children: BerNode[] }> {
  const messages = parseAll(buf);
  return messages
    .filter((m) => m.tag === 0x30)
    .map((m) => {
      const id = intValue(m.children[0]);
      const op = m.children[1];
      const resultCode = op?.children[0] ? intValue(op.children[0]) : undefined;
      return { id, tag: op?.tag ?? 0, resultCode, children: op?.children ?? [] };
    });
}

export function parseSearchGroups(buf: Buffer, groupAttr: string): string[] {
  const groups: string[] = [];
  const want = groupAttr.toLowerCase();
  for (const msg of parseLdapMessages(buf)) {
    if (msg.tag !== 0x64) continue;
    const attrs = msg.children[1];
    for (const seq of attrs?.children ?? []) {
      const name = textValue(seq.children[0]).toLowerCase();
      if (name !== want) continue;
      const set = seq.children[1];
      for (const val of set?.children ?? []) {
        const text = textValue(val).trim();
        if (text) groups.push(text);
      }
    }
  }
  return groups;
}

function defaultConnect(opts: { host: string; port: number; tls: boolean }, rejectUnauthorized: boolean): Promise<Socket> {
  return new Promise((resolve, reject) => {
    const onErr = (err: Error) => reject(err);
    if (opts.tls) {
      const sock = tlsConnect({ host: opts.host, port: opts.port, rejectUnauthorized, servername: opts.host });
      sock.once("secureConnect", () => {
        sock.off("error", onErr);
        resolve(sock);
      });
      sock.once("error", onErr);
      return;
    }
    const sock = netConnect({ host: opts.host, port: opts.port });
    sock.once("connect", () => {
      sock.off("error", onErr);
      resolve(sock);
    });
    sock.once("error", onErr);
  });
}

function readSocket(sock: Socket, timeoutMs: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("LDAP timed out"));
    }, timeoutMs);
    const onData = (c: Buffer) => {
      chunks.push(c);
      const buf = Buffer.concat(chunks);
      const parsed = parseBer(buf, 0);
      if (parsed && parsed.end <= buf.length) {
        // May be more than one message (search entry + done)
        if (buf.length >= parsed.end) {
          const msgs = parseLdapMessages(buf);
          const done = msgs.some((m) => m.tag === 0x61 || m.tag === 0x65);
          if (done || msgs.some((m) => m.tag === 0x61)) {
            cleanup();
            resolve(buf);
          }
        }
      }
    };
    const onErr = (err: Error) => {
      cleanup();
      reject(err);
    };
    const cleanup = () => {
      clearTimeout(timer);
      sock.off("data", onData);
      sock.off("error", onErr);
    };
    sock.on("data", onData);
    sock.once("error", onErr);
  });
}

function parseLdapUrl(raw: string): { host: string; port: number; tls: boolean } | undefined {
  try {
    const u = new URL(raw);
    if (u.protocol !== "ldaps:" && u.protocol !== "ldap:") return undefined;
    const tls = u.protocol === "ldaps:";
    const port = u.port ? Number(u.port) : tls ? 636 : 389;
    if (!u.hostname || !Number.isInteger(port)) return undefined;
    return { host: u.hostname, port, tls };
  } catch {
    return undefined;
  }
}

function substitute(template: string, username: string): string {
  return template.replaceAll("{username}", username);
}

function parseFilter(filter: string): { attr: string; value: string } | undefined {
  const m = /^\(\s*([^=()]+)=([^)]+)\)\s*$/.exec(filter.trim());
  if (!m) return undefined;
  return { attr: m[1]?.trim() ?? "", value: m[2]?.trim() ?? "" };
}

export function cnOf(dn: string): string {
  const m = /^cn=([^,]+)/i.exec(dn.trim());
  return (m?.[1] ?? dn).trim().toLowerCase();
}

export function groupMatches(got: string[], allow: string[]): boolean {
  if (!allow.length) return true;
  const want = allow.map((s) => s.trim().toLowerCase()).filter(Boolean);
  return got.some((g) => {
    const n = g.trim().toLowerCase();
    const cn = cnOf(g);
    return want.some((a) => n === a || cn === a || n.includes(a));
  });
}

async function writeAndRead(sock: Socket, packet: Buffer, timeoutMs: number): Promise<Buffer> {
  sock.write(packet);
  return readSocket(sock, timeoutMs);
}

export function createLdapAuthenticator(opts: LdapOptions): PasswordAuthenticator {
  const timeoutMs = 4000;
  return {
    method: "ldap",
    async login(creds: PasswordLogin) {
      const parsed = parseLdapUrl(opts.url);
      if (!parsed) return { ok: false, reason: "LDAP URL must be ldaps:// (preferred) or ldap://" };
      if (!parsed.tls) {
        return { ok: false, reason: "LDAP requires LDAPS (ldaps://). Plain ldap:// is refused." };
      }
      if (!creds.username.trim() || !creds.password) {
        return { ok: false, reason: "username and password required" };
      }
      const username = creds.username.trim();
      const connect =
        opts.connect ?? ((o) => defaultConnect(o, opts.tlsRejectUnauthorized));
      let sock: Socket;
      try {
        sock = await connect({ host: parsed.host, port: parsed.port, tls: true });
      } catch {
        return { ok: false, reason: "LDAPS is not reachable" };
      }
      const close = () => {
        try {
          sock.write(encodeLdapUnbind(99));
        } catch {
          /* ignore */
        }
        sock.destroy();
      };
      try {
        const userDn = substitute(opts.bindDnTemplate, username);
        if (opts.serviceDn && opts.servicePassword) {
          const svc = await writeAndRead(sock, encodeLdapBind(1, opts.serviceDn, opts.servicePassword), timeoutMs);
          const svcMsg = parseLdapMessages(svc)[0];
          if (!svcMsg || svcMsg.resultCode !== 0) return { ok: false, reason: "LDAP service bind failed" };
        }
        const bind = await writeAndRead(sock, encodeLdapBind(2, userDn, creds.password), timeoutMs);
        const bindMsg = parseLdapMessages(bind)[0];
        if (!bindMsg || bindMsg.resultCode !== 0) return { ok: false, reason: "LDAP rejected the user" };
        let groups: string[] = [];
        const parsedFilter = parseFilter(substitute(opts.filter || `(sAMAccountName={username})`, username));
        if (opts.baseDn && parsedFilter) {
          const search = encodeLdapSearch({
            messageId: 3,
            baseDn: opts.baseDn,
            filterAttr: parsedFilter.attr,
            filterValue: parsedFilter.value,
            attributes: [opts.groupAttr || "memberOf"],
          });
          const searchBuf = await writeAndRead(sock, search, timeoutMs);
          groups = parseSearchGroups(searchBuf, opts.groupAttr || "memberOf");
        }
        if (!groupMatches(groups, opts.allowedGroups)) {
          return { ok: false, reason: "LDAP group is not in the allowlist" };
        }
        return { ok: true, subject: username, groups };
      } catch {
        return { ok: false, reason: "LDAP bind failed" };
      } finally {
        close();
      }
    },
  };
}
