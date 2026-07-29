/**
 * Copy MediaPipe's wasm runtime out of node_modules into public/.
 *
 * FilesetResolver loads these at runtime from a URL. Serving them ourselves
 * rather than off a CDN keeps the app free of a third-party runtime dependency,
 * which matters when the thing is used on patchy range LTE.
 *
 * Runs automatically before dev and build. The copied files are generated
 * output, so they are gitignored.
 */

import { cp, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const dest = join(root, "public", "mediapipe", "wasm");

await mkdir(dirname(dest), { recursive: true });
await cp(src, dest, { recursive: true });
console.log(`copied MediaPipe wasm runtime -> public/mediapipe/wasm`);
