const MASK = "••••••••";

export function maskSecret(value: string | undefined): string | undefined {
  if (value === undefined || value === "") return value;
  if (value.startsWith("${") && value.endsWith("}")) return value;
  if (value === "ollama") return value;
  return MASK;
}

export function isMaskedSecret(value: unknown): boolean {
  return typeof value === "string" && (value === MASK || /^•+$/.test(value) || /^\*+$/.test(value));
}

export function redactHeaders(headers: Record<string, string> | undefined): Record<string, string> | undefined {
  if (!headers) return headers;
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    out[key] = /authorization|token|secret|key|password/i.test(key) ? (maskSecret(value) ?? MASK) : value;
  }
  return out;
}

export function restoreMaskedSecrets(current: unknown, incoming: unknown): unknown {
  if (isMaskedSecret(incoming)) return current;
  if (Array.isArray(incoming)) {
    const currentArr = Array.isArray(current) ? current : [];
    return incoming.map((item, index) => restoreMaskedSecrets(currentArr[index], item));
  }
  if (incoming && typeof incoming === "object") {
    const currentRec = current && typeof current === "object" && !Array.isArray(current)
      ? (current as Record<string, unknown>)
      : {};
    const out: Record<string, unknown> = { ...currentRec };
    for (const [key, nested] of Object.entries(incoming as Record<string, unknown>)) {
      if (nested === null) {
        delete out[key];
        continue;
      }
      out[key] = restoreMaskedSecrets(currentRec[key], nested);
    }
    return out;
  }
  return incoming;
}

export function redactConfigValue(value: unknown): unknown {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(redactConfigValue);
  if (!value || typeof value !== "object") return value;
  const record = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [key, nested] of Object.entries(record)) {
    if (key === "apiKey" && typeof nested === "string") {
      out[key] = maskSecret(nested);
    } else if (key === "headers" && nested && typeof nested === "object" && !Array.isArray(nested)) {
      out[key] = redactHeaders(nested as Record<string, string>);
    } else {
      out[key] = redactConfigValue(nested);
    }
  }
  return out;
}
