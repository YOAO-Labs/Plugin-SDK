import crypto from "node:crypto";
import fs from "node:fs";

export function readSecret(env, valueKey, fileKey) {
  const source = env || process.env;
  const direct = String(source[valueKey] || "").trim();
  const filePath = String(source[fileKey] || "").trim();
  if (direct && filePath) {
    throw new Error(`configure only one of ${valueKey} or ${fileKey}`);
  }
  if (filePath) {
    const content = fs.readFileSync(filePath, "utf8").trim();
    if (!content) {
      throw new Error(`${fileKey} is empty`);
    }
    return content;
  }
  if (!direct) {
    throw new Error(`${valueKey} or ${fileKey} is required`);
  }
  return direct;
}

export function verifyBearerToken(authorizationHeader, expectedToken) {
  if (typeof expectedToken !== "string" || !expectedToken) {
    return false;
  }
  if (typeof authorizationHeader !== "string" || !authorizationHeader.startsWith("Bearer ")) {
    return false;
  }
  const supplied = Buffer.from(authorizationHeader.slice("Bearer ".length), "utf8");
  const expected = Buffer.from(expectedToken, "utf8");
  if (supplied.length !== expected.length) {
    return false;
  }
  return crypto.timingSafeEqual(supplied, expected);
}
