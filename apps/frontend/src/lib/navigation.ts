const ALLOWED_NEXT_PATHS = new Set([
  "/films",
  "/discover",
  "/lists",
  "/admin",
  "/taste-profile",
]);
const NEXT_STORAGE_KEY = "filmfinder.next-path.v1";

export function safeNextPath(value: string | null): string | null {
  if (!value || !value.startsWith("/") || value.startsWith("//")) {
    return null;
  }
  const path = value.split("?")[0].split("#")[0];
  return ALLOWED_NEXT_PATHS.has(path) ? path : null;
}

export function readNextPath(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return safeNextPath(
    new URLSearchParams(window.location.search).get("next"),
  );
}

export function rememberNextPath(path: string | null): void {
  if (typeof window === "undefined" || !path) {
    return;
  }
  try {
    window.sessionStorage.setItem(NEXT_STORAGE_KEY, path);
  } catch {
    return;
  }
}

export function consumeStoredNextPath(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const stored = window.sessionStorage.getItem(NEXT_STORAGE_KEY);
    window.sessionStorage.removeItem(NEXT_STORAGE_KEY);
    return safeNextPath(stored);
  } catch {
    return null;
  }
}
