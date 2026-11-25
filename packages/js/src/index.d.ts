export const EXTENSION_MANIFEST_SCHEMA: "yoao-extension-manifest/v1";

export type ExtensionPoint =
  | "inference"
  | "notification_channel"
  | "format_adapter"
  | "trainer"
  | "theme";

export interface ExtensionManifestV1 {
  manifest_schema: "yoao-extension-manifest/v1";
  extension_point: ExtensionPoint;
  plugin_id: string;
  version: string;
  display_name?: string;
  description?: string;
  homepage?: string;
  publisher: string;
  license: string;
  protocol_version: string;
  entrypoint?: string[];
  capabilities: string[];
  resource?: { memory_mb: number };
  extension?: Record<string, unknown>;
  artifact?: string;
}

export interface ThemeTokensDocument {
  light?: Record<string, string>;
  dark?: Record<string, string>;
}

export function validateManifest(manifest: unknown): ExtensionManifestV1;
export function validateThemeTokens(themeDoc: unknown): ThemeTokensDocument;
export function readSecret(
  env: Record<string, string | undefined> | undefined,
  valueKey: string,
  fileKey: string,
): string;
export function verifyBearerToken(
  authorizationHeader: string | undefined,
  expectedToken: string,
): boolean;
