import { describe, expect, it } from "vitest";
import { formatBytes, validateZipFile } from "@/lib/upload";

function fileWithSize(name: string, size: number): File {
  const file = new File(["content"], name, { type: "application/zip" });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

describe("validateZipFile", () => {
  it("accepts a non-empty .zip", () => {
    expect(validateZipFile(fileWithSize("export.ZIP", 1024))).toEqual({
      ok: true,
    });
  });

  it("rejects other extensions", () => {
    const result = validateZipFile(fileWithSize("export.rar", 1024));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.message).toContain(".zip");
    }
  });

  it("rejects empty and oversized files", () => {
    expect(validateZipFile(fileWithSize("export.zip", 0)).ok).toBe(false);
    const oversized = validateZipFile(fileWithSize("export.zip", 21 * 1024 * 1024));
    expect(oversized.ok).toBe(false);
    if (!oversized.ok) {
      expect(oversized.message).toContain("larger than");
    }
  });
});

describe("formatBytes", () => {
  it("formats bytes, kilobytes and megabytes", () => {
    expect(formatBytes(400)).toBe("400 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });
});
