const TOKEN_NAME_RE = /^--sc-[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SAFE_COLOR_OR_VALUE_RE =
  /^(?:#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla|oklch)\([0-9a-zA-Z.,%/\s+-]{1,80}\)|[0-9]{1,4}(?:\.[0-9]{1,4})?(?:px|rem|em|%)|0)$/;

const FORBIDDEN_VALUE_SUBSTRINGS = ["url(", "expression(", "javascript:", ";", "{", "}", "<", ">", "@import"];

export function validateThemeTokens(themeDoc) {
  if (!themeDoc || typeof themeDoc !== "object" || Array.isArray(themeDoc)) {
    throw new Error("theme tokens document must be a JSON object");
  }
  const modes = Object.keys(themeDoc);
  if (modes.length === 0) {
    throw new Error("theme tokens document must define at least one mode (light or dark)");
  }
  for (const mode of modes) {
    if (mode !== "light" && mode !== "dark") {
      throw new Error(`unsupported theme mode: ${mode}`);
    }
    const tokens = themeDoc[mode];
    if (!tokens || typeof tokens !== "object" || Array.isArray(tokens)) {
      throw new Error(`theme mode ${mode} must be an object of --sc-* tokens`);
    }
    const entries = Object.entries(tokens);
    if (entries.length === 0 || entries.length > 256) {
      throw new Error(`theme mode ${mode} must define 1..256 tokens`);
    }
    for (const [name, value] of entries) {
      if (!TOKEN_NAME_RE.test(name)) {
        throw new Error(`invalid semantic token name: ${name} (must start with --sc-)`);
      }
      if (typeof value !== "string" || !value.trim() || value.length > 128) {
        throw new Error(`invalid value for token ${name}`);
      }
      const lowered = value.toLowerCase();
      if (FORBIDDEN_VALUE_SUBSTRINGS.some((bad) => lowered.includes(bad))) {
        throw new Error(`forbidden CSS construct in token ${name}`);
      }
      if (!SAFE_COLOR_OR_VALUE_RE.test(value.trim())) {
        throw new Error(`unsupported CSS value for token ${name}: ${value}`);
      }
    }
  }
  return themeDoc;
}
