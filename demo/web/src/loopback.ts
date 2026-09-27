/** Return whether `host` is a loopback name or address. */
export function isLoopbackHost(host: string): boolean {
  const value = host.trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (value === "localhost" || value === "::1") {
    return true;
  }
  const parts = value.split(".");
  if (parts.length !== 4 || parts.some((part) => !/^\d{1,3}$/.test(part))) {
    return false;
  }
  const numbers = parts.map((part) => Number(part));
  return numbers[0] === 127 && numbers.every((number) => number <= 255);
}
