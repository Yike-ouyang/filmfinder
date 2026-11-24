import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearTasteProfileDraft,
  loadTasteProfileDraft,
  saveTasteProfileDraft,
} from "@/lib/tasteProfileDraft";
import type { TasteProfileAnswer } from "@/types/tasteProfile";

function answers(): TasteProfileAnswer[] {
  return Array.from({ length: 20 }, (_, index) => ({
    internal_movie_id: 1001 + index,
    rating: index < 3 ? 9 : null,
  }));
}

beforeEach(() => {
  window.sessionStorage.clear();
});

afterEach(() => {
  window.sessionStorage.clear();
});

describe("taste profile draft", () => {
  it("round-trips 1-10 and haven't seen answers", () => {
    const value = answers();
    expect(saveTasteProfileDraft(value)).toBe(true);
    expect(loadTasteProfileDraft()).toEqual(value);
  });

  it("ignores the previous draft format", () => {
    window.sessionStorage.setItem(
      "filmfinder.taste-profile-draft.v1",
      JSON.stringify({
        version: 1,
        savedAt: Date.now(),
        answers: answers(),
      }),
    );
    expect(loadTasteProfileDraft()).toBeNull();
  });

  it("rejects out-of-range ratings and duplicates", () => {
    const invalid = answers();
    invalid[0] = { ...invalid[0], rating: 11 };
    saveTasteProfileDraft(invalid);
    expect(loadTasteProfileDraft()).toBeNull();

    const duplicate = answers();
    duplicate[1] = {
      ...duplicate[1],
      internal_movie_id: duplicate[0].internal_movie_id,
    };
    saveTasteProfileDraft(duplicate);
    expect(loadTasteProfileDraft()).toBeNull();
  });

  it("clears the draft", () => {
    saveTasteProfileDraft(answers());
    clearTasteProfileDraft();
    expect(loadTasteProfileDraft()).toBeNull();
  });
});
