"use client";

import Link from "next/link";
import { LoaderCircle, RotateCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { DiscoverExperience } from "@/components/discover/DiscoverExperience";
import { EmptyDiscovery } from "@/components/discover/EmptyDiscovery";
import { ApiError, errorMessage } from "@/lib/api";
import {
  fetchCurrentRecommendations,
  requestRecommendationRefresh,
  type CurrentRecommendations,
} from "@/lib/serverRecommendations";
import { undoSwipe, saveSwipe, decisionToAction } from "@/lib/swipes";
import {
  loadBatch,
  saveBatch,
  type BatchLoadResult,
} from "@/lib/storage";
import type {
  DecisionKind,
  Recommendation,
  RecommendationBatch,
  RequestMoreResult,
} from "@/types/recommendations";

const POLL_DELAY_MS = 1500;
const POLL_ATTEMPTS = 40;

function DiscoverLoading() {
  return (
    <main
      id="main"
      className="relative flex min-h-svh items-center justify-center bg-background px-6"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(90%_70%_at_50%_0%,#1a1a1a_0%,#0c0c0c_45%,#080808_80%)]"
      />
      <div
        role="status"
        aria-label="Loading recommendations"
        className="relative z-10 flex flex-col items-center gap-4 text-white/55"
      >
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin text-white/60"
        />
        <p className="text-[10px] uppercase tracking-[0.28em]">
          Preparing discoveries
        </p>
      </div>
    </main>
  );
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function buildServerBatch(
  current: CurrentRecommendations,
): RecommendationBatch | null {
  if (!current.response || current.batchId === null) {
    return null;
  }
  const saved = loadBatch();
  const decisions =
    saved.status === "ready" &&
    saved.batch?.batchId === current.batchId
      ? saved.batch.decisions
      : [];
  const batch: RecommendationBatch = {
    version: 1,
    batchId: current.batchId,
    savedAt: Date.now(),
    response: current.response,
    decisions,
  };
  saveBatch(
    batch.response,
    batch.decisions,
    batch.savedAt,
    batch.batchId,
  );
  return batch;
}

function recommendationIds(batch: RecommendationBatch): Set<number> {
  return new Set(
    batch.response.recommendations.map(
      (recommendation) => recommendation.internal_movie_id,
    ),
  );
}

export function DiscoverPageContent() {
  const { user, loading: authLoading } = useAuth();
  const [result, setResult] = useState<BatchLoadResult | null>(null);
  const [serverBatch, setServerBatch] =
    useState<RecommendationBatch | null>(null);
  const [serverUserId, setServerUserId] = useState<number | null>(null);
  const [serverLoading, setServerLoading] = useState(true);
  const [serverError, setServerError] = useState<string | null>(null);

  const persistDecision = useCallback(
    async (kind: DecisionKind, recommendation: Recommendation) => {
      await saveSwipe(
        recommendation.internal_movie_id,
        decisionToAction(kind),
      );
    },
    [],
  );

  const persistUndo = useCallback(
    async (recommendation: Recommendation) => {
      try {
        await undoSwipe(recommendation.internal_movie_id);
      } catch (error) {
        if (error instanceof ApiError && error.code === "swipe_not_found") {
          return;
        }
        throw error;
      }
    },
    [],
  );

  const pollForNewBatch = useCallback(
    async (
      excludeBatchId: number | null,
    ): Promise<CurrentRecommendations | null> => {
      for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
        await delay(POLL_DELAY_MS);
        try {
          const current = await fetchCurrentRecommendations();
          if (current.status === "failed") {
            throw new ApiError(
              "generation_failed",
              200,
              "Recommendations could not be generated.",
            );
          }
          if (
            current.response &&
            current.batchId !== null &&
            current.batchId !== excludeBatchId
          ) {
            return current;
          }
        } catch (error) {
          if (
            !(error instanceof ApiError) ||
            error.code !== "no_recommendations"
          ) {
            throw error;
          }
        }
      }
      return null;
    },
    [],
  );

  const loadServerBatch = useCallback(async () => {
    const userId = user?.id;
    if (userId === undefined) {
      return;
    }
    setServerLoading(true);
    setServerError(null);
    try {
      let current = await fetchCurrentRecommendations();
      if (current.status === "failed") {
        throw new ApiError(
          "generation_failed",
          200,
          "Recommendations could not be generated. Import a profile or try again.",
        );
      }
      if (current.status === "pending") {
        const ready = await pollForNewBatch(null);
        if (!ready) {
          throw new ApiError(
            "backend_unavailable",
            null,
            "The recommendation calculation is taking too long.",
          );
        }
        current = ready;
      }
      const batch = buildServerBatch(current);
      if (!batch) {
        throw new ApiError("invalid_response", 200);
      }
      setServerUserId(userId);
      setServerBatch(batch);
    } catch (error) {
      if (error instanceof ApiError && error.code === "no_recommendations") {
        try {
          const status = await requestRecommendationRefresh();
          if (status === "pending") {
            const ready = await pollForNewBatch(null);
            const batch = ready ? buildServerBatch(ready) : null;
            if (batch) {
              setServerUserId(userId);
              setServerBatch(batch);
              return;
            }
          }
        } catch (refreshError) {
          setServerError(errorMessage(refreshError));
          return;
        }
      }
      setServerError(errorMessage(error));
    } finally {
      setServerLoading(false);
    }
  }, [pollForNewBatch, user]);

  const requestMore = useCallback(async (): Promise<RequestMoreResult> => {
    if (!serverBatch?.batchId) {
      return {
        status: "error",
        message: "Recommendations are not ready yet.",
      };
    }
    try {
      const status = await requestRecommendationRefresh();
      if (status === "no_new_candidates") {
        return { status: "no_new" };
      }
      const next = await pollForNewBatch(serverBatch.batchId);
      if (!next) {
        return {
          status: "error",
          message: "New recommendations are taking too long. Try again.",
        };
      }
      if (!next.response) {
        return {
          status: "error",
          message: "The new recommendations could not be loaded.",
        };
      }
      const previousIds = recommendationIds(serverBatch);
      const hasNewFilm = next.response.recommendations.some(
        (recommendation) =>
          !previousIds.has(recommendation.internal_movie_id),
      );
      if (!hasNewFilm) {
        return { status: "no_new" };
      }
      const nextBatch = buildServerBatch(next);
      if (!nextBatch) {
        return {
          status: "error",
          message: "The new recommendations could not be loaded.",
        };
      }
      setServerBatch(nextBatch);
      return { status: "ready" };
    } catch (error) {
      return { status: "error", message: errorMessage(error) };
    }
  }, [pollForNewBatch, serverBatch]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (!cancelled) {
        setResult(loadBatch());
      }
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!user) {
      return;
    }
    const timer = window.setTimeout(() => {
      void loadServerBatch();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [user, loadServerBatch]);

  if (authLoading) {
    return <DiscoverLoading />;
  }

  if (user) {
    if (
      !serverBatch ||
      serverUserId !== user.id ||
      (serverLoading && !serverBatch)
    ) {
      if (serverError) {
        return (
          <main
            id="main"
            className="relative flex min-h-svh items-center justify-center overflow-hidden bg-background px-6"
          >
            <div
              aria-hidden="true"
              className="absolute inset-0 bg-[radial-gradient(100%_80%_at_50%_0%,#1b1b1b_0%,#0d0d0d_45%,#080808_80%)]"
            />
            <div className="relative z-10 w-full max-w-md text-center">
              <h1 className="text-3xl font-semibold tracking-[-0.035em] text-white">
                Recommendations unavailable
              </h1>
              <p className="mt-4 text-sm leading-relaxed text-white/55">
                {serverError}
              </p>
              <div className="mt-7 flex flex-col gap-3">
                <button
                  type="button"
                  onClick={() => void loadServerBatch()}
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90"
                >
                  <RotateCw aria-hidden="true" className="h-4 w-4" />
                  Try again
                </button>
                <Link
                  href="/"
                  className="inline-flex h-11 items-center justify-center rounded-full border border-white/10 text-sm font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
                >
                  Import Letterboxd
                </Link>
              </div>
            </div>
          </main>
        );
      }
      return <DiscoverLoading />;
    }
    return (
      <DiscoverExperience
        initialBatch={serverBatch}
        persistDecision={persistDecision}
        persistUndo={persistUndo}
        onRequestMore={requestMore}
        key={serverBatch.batchId ?? serverBatch.savedAt}
      />
    );
  }

  if (!result) {
    return <DiscoverLoading />;
  }

  if (result.status !== "ready" || !result.batch) {
    return (
      <EmptyDiscovery
        status={result.status === "ready" ? "missing" : result.status}
      />
    );
  }

  return (
    <DiscoverExperience
      initialBatch={result.batch}
      key={result.batch.savedAt}
    />
  );
}
