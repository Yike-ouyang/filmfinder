import { ApiError } from "@/lib/api";
import { authenticatedRequest } from "@/lib/auth";
import type { DecisionKind } from "@/types/recommendations";
import type {
  SwipeAction,
  SwipeItem,
  SwipeLists,
  SwipeMutation,
  SwipeUndoResult,
} from "@/types/swipes";

const SWIPE_ACTIONS: readonly SwipeAction[] = [
  "interested",
  "not_interested",
  "already_seen",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isSwipeAction(value: unknown): value is SwipeAction {
  return (
    typeof value === "string" &&
    SWIPE_ACTIONS.includes(value as SwipeAction)
  );
}

function asNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asNullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function normalizeSwipe(value: unknown): SwipeItem {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  const internalId = value.internal_movie_id;
  const movieId = value.movie_id;
  const title = value.title;
  const action = value.action;
  const updatedAt = value.updated_at;
  if (
    typeof internalId !== "number" ||
    (typeof movieId !== "string" && typeof movieId !== "number") ||
    typeof title !== "string" ||
    !isSwipeAction(action) ||
    typeof updatedAt !== "string"
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    internal_movie_id: internalId,
    movie_id: String(movieId),
    title,
    year: asNullableNumber(value.year),
    genres: Array.isArray(value.genres)
      ? value.genres.filter(
          (genre): genre is string => typeof genre === "string",
        )
      : [],
    tmdb_id: asNullableNumber(value.tmdb_id),
    poster_path: asNullableString(value.poster_path),
    backdrop_path: asNullableString(value.backdrop_path),
    action,
    updated_at: updatedAt,
  };
}

function normalizeSwipeList(value: unknown): SwipeItem[] {
  if (!Array.isArray(value)) {
    throw new ApiError("invalid_response", 200);
  }
  return value.map(normalizeSwipe);
}

export function decisionToAction(kind: DecisionKind): SwipeAction {
  if (kind === "interested") {
    return "interested";
  }
  if (kind === "passed") {
    return "not_interested";
  }
  return "already_seen";
}

export async function fetchSwipeLists(): Promise<SwipeLists> {
  const payload = await authenticatedRequest("/api/v1/swipes");
  if (!isRecord(payload) || !isRecord(payload.counts)) {
    throw new ApiError("invalid_response", 200);
  }
  const interested = normalizeSwipeList(payload.interested);
  const notInterested = normalizeSwipeList(payload.not_interested);
  const counts = payload.counts;
  if (
    typeof counts.interested !== "number" ||
    typeof counts.not_interested !== "number"
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    interested,
    not_interested: notInterested,
    counts: {
      interested: counts.interested,
      not_interested: counts.not_interested,
    },
  };
}

export async function saveSwipe(
  internalMovieId: number,
  action: SwipeAction,
): Promise<SwipeMutation> {
  const payload = await authenticatedRequest(
    `/api/v1/swipes/${internalMovieId}`,
    {
      method: "PUT",
      body: JSON.stringify({ action }),
    },
  );
  if (
    !isRecord(payload) ||
    typeof payload.event_id !== "number"
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    swipe: normalizeSwipe(payload.swipe),
    event_id: payload.event_id,
  };
}

export async function undoSwipe(
  internalMovieId: number,
): Promise<SwipeUndoResult> {
  const payload = await authenticatedRequest(
    `/api/v1/swipes/${internalMovieId}/undo`,
    { method: "POST" },
  );
  if (!isRecord(payload)) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    swipe:
      payload.swipe === null || payload.swipe === undefined
        ? null
        : normalizeSwipe(payload.swipe),
  };
}

export async function removeSwipe(
  internalMovieId: number,
): Promise<void> {
  await authenticatedRequest(`/api/v1/swipes/${internalMovieId}`, {
    method: "DELETE",
  });
}
