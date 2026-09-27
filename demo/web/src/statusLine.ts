/** Status copy for the configured API origin. */
export function localStatusLine(apiOrigin: string): string {
  const host = new URL(apiOrigin).hostname;
  return `Local · ${host} · no cloud PHI`;
}
