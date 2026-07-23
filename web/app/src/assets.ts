/** Heavyweight immutable payloads (trained nets, solver tables) live in the
 * arcade-assets R2 bucket under content-addressed names, recorded in
 * asset-manifest.json (scripts/r2-assets.mjs); the build bakes their URLs
 * into `__ARCADE_ASSETS__`. The fallback covers anything still served from
 * public/ (fixtures, the doom payloads). */
export function assetUrl(path: string): string {
  return __ARCADE_ASSETS__[path] ?? `${import.meta.env.BASE_URL}${path}`;
}
