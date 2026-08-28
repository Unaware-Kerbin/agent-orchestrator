export type McpAuthMethod = "local-token" | "ldap" | "radius";

export type McpPrincipal = {
  method: McpAuthMethod | "session";
  subject: string;
  groups: string[];
};

export type PasswordLogin = {
  username: string;
  password: string;
};

export type McpCredentials =
  | { kind: "bearer"; token: string }
  | { kind: "basic"; username: string; password: string }
  | { kind: "none" };

export type PasswordAuthenticator = {
  method: "ldap" | "radius";
  login: (creds: PasswordLogin) => Promise<{ ok: true; groups: string[]; subject: string } | { ok: false; reason: string }>;
};

export type LocalTokenAuthenticator = {
  method: "local-token";
  token: string;
};

export type McpAuthConfig = {
  methods: McpAuthMethod[];
  localToken?: string;
  ldap?: PasswordAuthenticator;
  radius?: PasswordAuthenticator;
  ldapAllowedGroups: string[];
  radiusAllowedFilterIds: string[];
  sessionTtlMs: number;
};
