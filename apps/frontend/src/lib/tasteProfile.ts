import type {
  TasteProfileAnswer,
  TasteProfileMovie,
  TasteProfileRating,
} from "@/types/tasteProfile";

export const RATING_VALUES: readonly TasteProfileRating[] = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
];

export interface TasteProfileState {
  index: number;
  answers: TasteProfileAnswer[];
  started: boolean;
}

export type TasteProfileEvent =
  | { type: "start" }
  | {
      type: "answer";
      movie: TasteProfileMovie;
      rating: TasteProfileRating | null;
    }
  | { type: "undo" }
  | { type: "restart" };

export const initialTasteProfileState: TasteProfileState = {
  index: 0,
  answers: [],
  started: false,
};

export function tasteProfileReducer(
  state: TasteProfileState,
  event: TasteProfileEvent,
): TasteProfileState {
  switch (event.type) {
    case "start":
      return { ...state, started: true };
    case "restart":
      return { ...initialTasteProfileState, started: true };
    case "answer": {
      if (!state.started) {
        return state;
      }
      if (
        state.answers.some(
          (answer) =>
            answer.internal_movie_id === event.movie.internal_movie_id,
        )
      ) {
        return state;
      }
      const answers = state.answers.slice(0, state.index);
      answers[state.index] = {
        internal_movie_id: event.movie.internal_movie_id,
        rating: event.rating,
      };
      return {
        ...state,
        answers,
        index: state.index + 1,
      };
    }
    case "undo": {
      if (!state.started || state.index === 0) {
        return state;
      }
      const answers = state.answers.slice(0, state.index - 1);
      return {
        ...state,
        answers,
        index: state.index - 1,
      };
    }
    default:
      return state;
  }
}

export function countRatedAnswers(answers: TasteProfileAnswer[]): number {
  return answers.filter((answer) => answer.rating !== null).length;
}

export function isTasteProfileComplete(
  state: TasteProfileState,
  total: number,
): boolean {
  return state.started && state.index >= total;
}

export function hasEnoughRatings(
  answers: TasteProfileAnswer[],
  minimum: number,
): boolean {
  return countRatedAnswers(answers) >= Math.max(1, minimum);
}

export function ratingForMovie(
  answers: TasteProfileAnswer[],
  movie: TasteProfileMovie,
): TasteProfileRating | null {
  return (
    answers.find(
      (answer) =>
        answer.internal_movie_id === movie.internal_movie_id,
    )?.rating ?? null
  );
}
