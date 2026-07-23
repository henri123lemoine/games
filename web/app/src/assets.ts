/** Heavyweight immutable payloads (trained nets, solver tables) are published
 * to the arcade-assets R2 bucket under content-addressed names and pruned from
 * the deployed bundle (scripts/r2-assets.mjs); production builds bake their
 * URLs into `__ARCADE_ASSETS__`. Dev builds leave the map empty, so the same
 * files are served from public/ and everything works offline. */
export function assetUrl(path: string): string {
  return __ARCADE_ASSETS__[path] ?? `${import.meta.env.BASE_URL}${path}`;
}
