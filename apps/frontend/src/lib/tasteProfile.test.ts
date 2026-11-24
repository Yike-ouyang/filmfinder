import { describe, expect, it } from "vitest";
import {
  countRatedAnswers,
  hasEnoughRatings,
  initialTasteProfileState,
  isTasteProfileComplete,
  RATING_VALUES,
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
  user_rating: null,
};

describe("taste profile state", () => {
  it("exposes the full 1-10 rating scale", () => {
    expect(RATING_VALUES).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
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

  it("records answers, skips haven't seen and advances", () => {
    let state = tasteProfileReducer(initialTasteProfileState, {
      type: "start",
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 9,
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie: { ...movie, internal_movie_id: 2 },
      rating: null,
    });
    expect(state.index).toBe(2);
    expect(state.answers).toEqual([
      { internal_movie_id: 1, rating: 9 },
      { internal_movie_id: 2, rating: null },
    ]);
    expect(countRatedAnswers(state.answers)).toBe(1);
    expect(hasEnoughRatings(state.answers, 3)).toBe(false);
  });

  it("ignores duplicate answers from rapid clicks", () => {
    let state = tasteProfileReducer(initialTasteProfileState, {
      type: "start",
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 9,
    });
    const duplicate = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 4,
    });
    expect(duplicate).toBe(state);
    expect(duplicate.answers).toEqual([
      { internal_movie_id: 1, rating: 9 },
    ]);
    expect(duplicate.index).toBe(1);
  });

  it("undoes, replaces and restarts", () => {
    let state = tasteProfileReducer(initialTasteProfileState, {
      type: "start",
    });
    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 9,
    });
    state = tasteProfileReducer(state, { type: "undo" });
    expect(state.index).toBe(0);
    expect(state.answers).toEqual([]);

    state = tasteProfileReducer(state, {
      type: "answer",
      movie,
      rating: 1,
    });
    expect(state.answers).toEqual([
      { internal_movie_id: 1, rating: 1 },
    ]);
    state = tasteProfileReducer(state, { type: "restart" });
    expect(state).toEqual({ index: 0, answers: [], started: true });
  });

  it("detects completion and the minimum number of ratings", () => {
    const answers: TasteProfileAnswer[] = [
      { internal_movie_id: 1, rating: 9 },
      { internal_movie_id: 2, rating: null },
      { internal_movie_id: 3, rating: 5 },
      { internal_movie_id: 4, rating: 1 },
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
