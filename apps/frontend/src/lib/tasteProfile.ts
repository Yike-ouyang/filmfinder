import type {
  TasteProfileAction,
  TasteProfileAnswer,
  TasteProfileMovie,
  TasteProfileRating,
} from "@/types/tasteProfile";

export interface TasteActionDefinition {
  action: TasteProfileAction;
  label: string;
  shortcut: string;
  rating: TasteProfileRating | null;
  hint: string;
}

export const TASTE_ACTIONS: readonly TasteActionDefinition[] = [
  {
    action: "loved",
    label: "Loved it",
    shortcut: "1",
    rating: 8,
    hint: "Add to my favorites",
  },
  {
    action: "okay",
    label: "It was okay",
    shortcut: "2",
    rating: 5,
    hint: "A neutral rating",
  },
  {
    action: "disliked",
    label: "Didn't like it",
    shortcut: "3",
    rating: 3,
    hint: "Not for me",
  },
  {
    action: "no-opinion",
    label: "No opinion",
    shortcut: "4",
    rating: null,
    hint: "Skip without rating",
  },
] as const;

export const DEFAULT_MIN_TASTE_RATINGS = 3;

export interface TasteProfileState {
  index: number;
  answers: TasteProfileAnswer[];
  started: boolean;
}

export type TasteProfileEvent =
  | { type: "start" }
  | { type: "answer"; movie: TasteProfileMovie; rating: TasteProfileRating | null }
  | { type: "undo" }
  | { type: "restart" };

export const initialTasteProfileState: TasteProfileState = {
  index: 0,
  answers: [],
  started: false,
};

export function ratingForAction(
  action: TasteProfileAction,
): TasteProfileRating | null {
  const definition = TASTE_ACTIONS.find((item) => item.action === action);
  return definition ? definition.rating : null;
}

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
