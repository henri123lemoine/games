/** Read a required positive integer that crosses a wasm `u32` boundary.
 * Client-bot configuration is resolved before the factory runs, so a missing
 * or malformed value is a wiring error and must not silently become a default. */
export function requiredU32(opts: Record<string, string>, key: string): number {
  const raw = opts[key];
  const value = Number(raw);
  if (!raw || !Number.isInteger(value) || value <= 0 || value > 0xffff_ffff)
    throw new Error(`client bot requires ${key}=1..4294967295, got '${raw ?? ""}'`);
  return value;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
