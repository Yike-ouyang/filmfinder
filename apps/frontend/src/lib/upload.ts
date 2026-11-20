import { MAX_UPLOAD_BYTES, MAX_UPLOAD_SIZE_MB } from "@/lib/constants";

export type UploadValidation =
  | { ok: true }
  | { ok: false; message: string };

export function validateZipFile(
  file: File,
  maxBytes = MAX_UPLOAD_BYTES,
): UploadValidation {
  if (!file) {
    return { ok: false, message: "Choose a Letterboxd ZIP export first." };
  }
  if (file.size === 0) {
    return { ok: false, message: "This file is empty." };
  }
  if (!file.name.toLowerCase().endsWith(".zip")) {
    return {
      ok: false,
      message: "Please choose a .zip file exported from Letterboxd.",
    };
  }
  if (file.size > maxBytes) {
    return {
      ok: false,
      message: `This archive is larger than the ${formatBytes(maxBytes)} limit.`,
    };
  }
  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function uploadLimitLabel(): string {
  return `${MAX_UPLOAD_SIZE_MB} MB`;
}
