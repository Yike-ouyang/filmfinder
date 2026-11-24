import type {
  TasteProfileAnswer,
  TasteProfileRating,
} from "@/types/tasteProfile";

const DRAFT_KEY = "filmfinder.taste-profile-draft.v2";
const DRAFT_VERSION = 2;
const EXPECTED_ANSWERS = 20;

function getStorage(): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function isRating(value: unknown): value is TasteProfileRating | null {
  if (value === null) {
    return true;
  }
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= 10
  );
}

function isAnswer(value: unknown): value is TasteProfileAnswer {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.internal_movie_id === "number" &&
    Number.isInteger(candidate.internal_movie_id) &&
    candidate.internal_movie_id >= 0 &&
    isRating(candidate.rating)
  );
}

export function saveTasteProfileDraft(
  answers: TasteProfileAnswer[],
  savedAt = Date.now(),
): boolean {
  const storage = getStorage();
  if (!storage) {
    return false;
  }
  try {
    storage.setItem(
      DRAFT_KEY,
      JSON.stringify({
        version: DRAFT_VERSION,
        savedAt,
        answers,
      }),
    );
    return true;
  } catch {
    return false;
  }
}

export function loadTasteProfileDraft(): TasteProfileAnswer[] | null {
  const storage = getStorage();
  if (!storage) {
    return null;
  }
  let raw: string | null;
  try {
    raw = storage.getItem(DRAFT_KEY);
  } catch {
    return null;
  }
  if (!raw) {
    return null;
  }
  let payload: unknown;
  try {
    payload = JSON.parse(raw);
  } catch {
    clearTasteProfileDraft();
    return null;
  }
  if (typeof payload !== "object" || payload === null) {
    clearTasteProfileDraft();
    return null;
  }
  const candidate = payload as Record<string, unknown>;
  if (
    candidate.version !== DRAFT_VERSION ||
    typeof candidate.savedAt !== "number" ||
    !Array.isArray(candidate.answers) ||
    candidate.answers.length !== EXPECTED_ANSWERS ||
    !candidate.answers.every(isAnswer)
  ) {
    clearTasteProfileDraft();
    return null;
  }
  const identifiers = candidate.answers.map(
    (answer: TasteProfileAnswer) => answer.internal_movie_id,
  );
  if (new Set(identifiers).size !== identifiers.length) {
    clearTasteProfileDraft();
    return null;
  }
  return candidate.answers;
}

export function clearTasteProfileDraft(): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  try {
    storage.removeItem(DRAFT_KEY);
  } catch {
    return;
  }
}
