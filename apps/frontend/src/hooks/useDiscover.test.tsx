import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useDiscover } from "@/hooks/useDiscover";
import { makeBatch } from "@/test/fixtures";

describe("useDiscover", () => {
  it("starts on the first recommendation", () => {
    const { result } = renderHook(() => useDiscover(makeBatch(3)));

    expect(result.current.current?.title).toBe("Film One");
    expect(result.current.index).toBe(0);
    expect(result.current.total).toBe(3);
    expect(result.current.completed).toBe(false);
    expect(result.current.stats).toEqual({
      interested: 0,
      passed: 0,
      seen: 0,
    });
  });

  it("advances through decisions and counts each kind", () => {
    const { result } = renderHook(() => useDiscover(makeBatch(3)));

    act(() => result.current.decide("interested"));
    expect(result.current.index).toBe(1);
    expect(result.current.current?.title).toBe("Film Two");

    act(() => result.current.decide("passed"));
    expect(result.current.index).toBe(2);
    expect(result.current.current?.title).toBe("Film Three");

    act(() => result.current.decide("seen"));
    expect(result.current.completed).toBe(true);
    expect(result.current.current).toBeNull();
    expect(result.current.stats).toEqual({
      interested: 1,
      passed: 1,
      seen: 1,
    });
  });

  it("undoes the most recent decision", () => {
    const { result } = renderHook(() => useDiscover(makeBatch(3)));

    act(() => result.current.decide("passed"));
    act(() => result.current.decide("interested"));
    expect(result.current.index).toBe(2);

    act(() => result.current.undo());
    expect(result.current.index).toBe(1);
    expect(result.current.current?.title).toBe("Film Two");
    expect(result.current.stats.interested).toBe(0);
    expect(result.current.decisions).toHaveLength(1);

    act(() => result.current.undo());
    expect(result.current.index).toBe(0);
    expect(result.current.current?.title).toBe("Film One");
  });

  it("resumes from an existing decision prefix", () => {
    const batch = makeBatch(3, [
      { key: "1001", kind: "interested" },
      { key: "1002", kind: "passed" },
    ]);
    const { result } = renderHook(() => useDiscover(batch));

    expect(result.current.index).toBe(2);
    expect(result.current.current?.title).toBe("Film Three");
    expect(result.current.stats).toEqual({
      interested: 1,
      passed: 1,
      seen: 0,
    });
  });
});
