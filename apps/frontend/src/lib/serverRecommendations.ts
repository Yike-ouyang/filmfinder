import { ApiError, normalizeRecommendationResponse } from "@/lib/api";
import { authenticatedRequest } from "@/lib/auth";
import type { RecommendationResponse } from "@/types/recommendations";

export type CurrentRecommendationStatus =
  | "ready"
  | "refreshing"
  | "pending"
  | "failed";
export type RefreshStatus = "pending" | "no_new_candidates";

export interface CurrentRecommendations {
  status: CurrentRecommendationStatus;
  batchId: number | null;
  generatedAt: string | null;
  response: RecommendationResponse | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export async function fetchCurrentRecommendations(): Promise<CurrentRecommendations> {
  const payload = await authenticatedRequest(
    "/api/v1/recommendations/current",
  );
  if (!isRecord(payload)) {
    throw new ApiError("invalid_response", 200);
  }
  const status = payload.status;
  if (
    status !== "ready" &&
    status !== "refreshing" &&
    status !== "pending" &&
    status !== "failed"
  ) {
    throw new ApiError("invalid_response", 200);
  }
  const batchId =
    typeof payload.batch_id === "number" ? payload.batch_id : null;
  const generatedAt =
    typeof payload.generated_at === "string"
      ? payload.generated_at
      : null;
  const response =
    payload.response === null || payload.response === undefined
      ? null
      : normalizeRecommendationResponse(payload.response);
  if (
    status !== "pending" &&
    status !== "failed" &&
    (!batchId || !response)
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return { status, batchId, generatedAt, response };
}

export async function requestRecommendationRefresh(): Promise<RefreshStatus> {
  const payload = await authenticatedRequest(
    "/api/v1/recommendations/refresh",
    { method: "POST" },
  );
  if (
    !isRecord(payload) ||
    (payload.status !== "pending" &&
      payload.status !== "no_new_candidates")
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return payload.status;
}
