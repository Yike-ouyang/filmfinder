import { describe, expect, it } from "vitest";
import {
  countRatedAnswers,
  hasEnoughRatings,
  initialTasteProfileState,
  isTasteProfileComplete,
  ratingForAction,
  tasteProfileReducer,
} from "@/lib/tasteProfile";
import type {
  TasteProfileAnswer,
  TasteProfileMovie,
} from "@/types/tasteProfile";

const movie: TasteProfileMovie = {
  internal_movie_id: 1,
  movie_id: "film-one",
  title: "Film One",
  year: 2001,
  genres: ["Drama"],
  tmdb_id: 1,
  poster_path: "/poster.jpg",
  backdrop_path: "/backdrop.jpg",
};

describe("taste profile state", () => {
  it("maps the four actions to the expected ratings", () => {
    expect(ratingForAction("loved")).toBe(8);
    expect(ratingForAction("okay")).toBe(5);
    expect(ratingForAction("disliked")).toBe(3);
    expect(ratingForAction("no-opinion")).toBeNull();
  });

  it("starts only after the intro", () => {
    const started = tasteProfileReducer(initialTasteProfileState, {
      type: "start",
    });
    expect(started.started).toBe(true);
    const ignored = tasteProfileReducer(initialTasteProfileState, {
      type: "answer",
      movie,
      rating: 8,
    });
    expect(ignored).toBe(initialTasteProfileState);
  });

  it("records answers, skips no-opinion and advances", () => {
    let state = tasteProfileReducer(initialTasteProfileState, {
      type: "start",
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 8,
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie: { ...movie, internal_movie_id: 2 },
      rating: null,
    });
    expect(state.index).toBe(2);
    expect(state.answers).toEqual([
      { internal_movie_id: 1, rating: 8 },
      { internal_movie_id: 2, rating: null },
    ]);
    expect(countRatedAnswers(state.answers)).toBe(1);
    expect(hasEnoughRatings(state.answers, 3)).toBe(false);
  });

  it("undoes, replaces and restarts", () => {
    let state = tasteProfileReducer(initialTasteProfileState, {
      type: "start",
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 8,
    });
    state = tasteProfileReducer(state, { type: "undo" });
    expect(state.index).toBe(0);
    expect(state.answers).toEqual([]);

    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 3,
    });
    expect(state.answers).toEqual([
      { internal_movie_id: 1, rating: 3 },
    ]);
    state = tasteProfileReducer(state, { type: "restart" });
    expect(state).toEqual({ index: 0, answers: [], started: true });
  });

  it("detects completion and the minimum number of ratings", () => {
    const answers: TasteProfileAnswer[] = [
      { internal_movie_id: 1, rating: 8 },
      { internal_movie_id: 2, rating: null },
      { internal_movie_id: 3, rating: 5 },
      { internal_movie_id: 4, rating: 3 },
    ];
    expect(countRatedAnswers(answers)).toBe(3);
    expect(hasEnoughRatings(answers, 3)).toBe(true);
    expect(
      isTasteProfileComplete(
        { index: 4, answers, started: true },
        4,
      ),
    ).toBe(true);
    expect(
      isTasteProfileComplete(
        { index: 4, answers, started: false },
        4,
      ),
    ).toBe(false);
  });
});
