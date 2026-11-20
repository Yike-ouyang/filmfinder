import {
  SESSION_TTL_MS,
  STORAGE_KEY,
} from "@/lib/constants";
import { normalizeRecommendationResponse } from "@/lib/api";
import { recommendationKey } from "@/lib/recommendations";
import type {
  BatchLoadStatus,
  Decision,
  DecisionKind,
  RecommendationBatch,
  RecommendationResponse,
} from "@/types/recommendations";

export interface BatchLoadResult {
  status: BatchLoadStatus;
  batch: RecommendationBatch | null;
}

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

function isDecision(value: unknown): value is Decision {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.key === "string" &&
    (candidate.kind === "interested" ||
      candidate.kind === "passed" ||
      candidate.kind === "seen")
  );
}

function validDecisionPrefix(
  response: RecommendationResponse,
  decisions: unknown[],
): Decision[] {
  const valid: Decision[] = [];
  for (let index = 0; index < decisions.length; index += 1) {
    const decision = decisions[index];
    const recommendation = response.recommendations[index];
    if (
      !isDecision(decision) ||
      !recommendation ||
      decision.key !== recommendationKey(recommendation)
    ) {
      break;
    }
    valid.push({ key: decision.key, kind: decision.kind as DecisionKind });
  }
  return valid;
}

function clearStoredBatch(): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  try {
    storage.removeItem(STORAGE_KEY);
  } catch {
    return;
  }
}

export function loadBatch(): BatchLoadResult {
  const storage = getStorage();
  if (!storage) {
    return { status: "unavailable", batch: null };
  }

  let raw: string | null;
  try {
    raw = storage.getItem(STORAGE_KEY);
  } catch {
    return { status: "unavailable", batch: null };
  }

  if (!raw) {
    return { status: "missing", batch: null };
  }

  let payload: unknown;
  try {
    payload = JSON.parse(raw);
  } catch {
    clearStoredBatch();
    return { status: "invalid", batch: null };
  }

  if (typeof payload !== "object" || payload === null) {
    clearStoredBatch();
    return { status: "invalid", batch: null };
  }

  const candidate = payload as Record<string, unknown>;
  if (candidate.version !== 1 || typeof candidate.savedAt !== "number") {
    clearStoredBatch();
    return { status: "invalid", batch: null };
  }

  if (Date.now() - candidate.savedAt > SESSION_TTL_MS) {
    clearStoredBatch();
    return { status: "expired", batch: null };
  }

  let response: RecommendationResponse;
  try {
    response = normalizeRecommendationResponse(candidate.response);
  } catch {
    clearStoredBatch();
    return { status: "invalid", batch: null };
  }

  const decisions = Array.isArray(candidate.decisions)
    ? validDecisionPrefix(response, candidate.decisions)
    : [];

  return {
    status: "ready",
    batch: {
      version: 1,
      savedAt: candidate.savedAt,
      response,
      decisions,
    },
  };
}

export function saveBatch(
  response: RecommendationResponse,
  decisions: Decision[] = [],
  savedAt = Date.now(),
): boolean {
  const storage = getStorage();
  if (!storage) {
    return false;
  }

  const payload: RecommendationBatch = {
    version: 1,
    savedAt,
    response,
    decisions,
  };

  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

export function updateDecisions(
  batch: RecommendationBatch,
  decisions: Decision[],
): boolean {
  return saveBatch(batch.response, decisions, batch.savedAt);
}

export function clearBatch(): void {
  clearStoredBatch();
}
