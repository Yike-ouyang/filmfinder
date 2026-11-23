import { describe, expect, it } from "vitest";
import { safeNextPath } from "@/lib/navigation";

describe("safeNextPath", () => {
  it("allows internal application paths only", () => {
    expect(safeNextPath("/films")).toBe("/films");
    expect(safeNextPath("/admin")).toBe("/admin");
    expect(safeNextPath("/discover?x=1")).toBe("/discover");
    expect(safeNextPath("https://example.com")).toBeNull();
    expect(safeNextPath("//example.com")).toBeNull();
    expect(safeNextPath("/films/../admin")).toBeNull();
  });
});
