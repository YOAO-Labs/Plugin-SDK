#!/usr/bin/env node
import fs from "node:fs";
import process from "node:process";

import { validateManifest } from "./manifest.mjs";
import { validateThemeTokens } from "./theme.mjs";

const [, , cmd, targetPath] = process.argv;

if (!cmd || !targetPath) {
  console.error("Usage: yoao-plugin-js <validate|validate-theme> <file.json>");
  process.exit(2);
}

try {
  const raw = JSON.parse(fs.readFileSync(targetPath, "utf8"));
  if (cmd === "validate") {
    const manifest = validateManifest(raw);
    console.log(
      JSON.stringify({
        valid: true,
        extension_point: manifest.extension_point,
        plugin_id: manifest.plugin_id,
      }),
    );
  } else if (cmd === "validate-theme") {
    const theme = validateThemeTokens(raw);
    console.log(JSON.stringify({ valid: true, modes: Object.keys(theme) }));
  } else {
    console.error(`Unknown command: ${cmd}`);
    process.exit(2);
  }
} catch (err) {
  console.error(`INVALID: ${err.message}`);
  process.exit(1);
}
