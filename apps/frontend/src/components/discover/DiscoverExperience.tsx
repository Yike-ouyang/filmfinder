"use client";

import { AnimatePresence } from "motion/react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { DiscoveryComplete } from "@/components/discover/DiscoveryComplete";
import { MovieBackdrop } from "@/components/discover/MovieBackdrop";
import { MovieCard } from "@/components/discover/MovieCard";
import { MovieDetails } from "@/components/discover/MovieDetails";
import { SwipeControls } from "@/components/discover/SwipeControls";
import { TmdbAttribution } from "@/components/ui/TmdbAttribution";
import { useDiscover } from "@/hooks/useDiscover";
import { usePrefetchImages } from "@/hooks/usePrefetchImages";
import { errorMessage } from "@/lib/api";
import { recommendationKey } from "@/lib/recommendations";
import { clearBatch } from "@/lib/storage";
import type {
  DecisionKind,
  Recommendation,
  RecommendationBatch,
  RequestMoreResult,
} from "@/types/recommendations";

interface DiscoverExperienceProps {
  initialBatch: RecommendationBatch;
  persistDecision?: (
    kind: DecisionKind,
    recommendation: Recommendation,
  ) => Promise<void>;
  persistUndo?: (recommendation: Recommendation) => Promise<void>;
  onRequestMore?: () => Promise<RequestMoreResult>;
}

export function DiscoverExperience({
  initialBatch,
  persistDecision,
  persistUndo,
  onRequestMore,
}: DiscoverExperienceProps) {
  const router = useRouter();
  const {
    batch,
    current,
    decisions,
    index,
    total,
    completed,
    lastRecommendation,
    stats,
    exitDirection,
    decide,
    undo,
  } = useDiscover(initialBatch);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef(false);

  usePrefetchImages(batch.response.recommendations, index);

  const handleDecision = useCallback(
    async (kind: DecisionKind) => {
      if (!current || busyRef.current) {
        return;
      }
      const recommendation = current;
      const title = recommendation.title;
      setError(null);
      if (persistDecision) {
        busyRef.current = true;
        setBusy(true);
        try {
          await persistDecision(kind, recommendation);
        } catch (decisionError) {
          setError(errorMessage(decisionError));
          return;
        } finally {
          busyRef.current = false;
          setBusy(false);
        }
      }
      decide(kind);
      if (kind === "interested") {
        setAnnouncement(`Interested in ${title}.`);
      } else if (kind === "passed") {
        setAnnouncement(`Passed on ${title}.`);
      } else {
        setAnnouncement(`Marked ${title} as already seen.`);
      }
    },
    [current, decide, persistDecision],
  );

  const handleUndo = useCallback(async () => {
    const lastDecision = decisions[decisions.length - 1];
    if (!lastDecision || busyRef.current) {
      return;
    }
    const restored =
      batch.response.recommendations[decisions.length - 1];
    setError(null);
    if (persistUndo && restored) {
      busyRef.current = true;
      setBusy(true);
      try {
        await persistUndo(restored);
      } catch (undoError) {
        setError(errorMessage(undoError));
        return;
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    }
    undo();
    setAnnouncement(
      restored ? `Restored ${restored.title}.` : "Restored the previous film.",
    );
  }, [
    batch.response.recommendations,
    decisions,
    persistUndo,
    undo,
  ]);

  const handleImportAnother = () => {
    clearBatch();
    router.push("/");
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        detailsOpen ||
        !current ||
        busyRef.current ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey
      ) {
        return;
      }
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        target.closest("input, textarea, select, [contenteditable='true']")
      ) {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        handleDecision("passed");
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        handleDecision("interested");
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        handleDecision("seen");
      } else if (event.key === "Backspace" && decisions.length > 0) {
        event.preventDefault();
        handleUndo();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [current, decisions.length, detailsOpen, handleDecision, handleUndo]);

  const activeRecommendation = current ?? lastRecommendation;

  return (
    <main
      id="main"
      className="relative flex min-h-svh flex-col overflow-hidden bg-background"
    >
      <MovieBackdrop
        recommendation={activeRecommendation}
        priority={index === 0}
      />

      <h1 className="sr-only">Discover films</h1>

      <div className="safe-bottom-tight relative z-10 flex min-h-svh flex-col px-5 pt-20 sm:px-8 sm:pt-24">
        <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col">
          <div className="flex h-8 shrink-0 items-center justify-between">
            <span aria-hidden="true" />
            {index === 0 && !completed ? (
              <p className="hidden text-[9px] uppercase tracking-[0.2em] text-white/40 sm:block">
                Swipe or use the controls
              </p>
            ) : (
              <span aria-hidden="true" />
            )}
          </div>

          <div className="flex flex-1 items-center justify-center py-3">
            <AnimatePresence mode="wait" custom={exitDirection}>
              {current ? (
                <MovieCard
                  key={recommendationKey(current)}
                  recommendation={current}
                  exitDirection={exitDirection}
                  onDecision={handleDecision}
                  onOpenDetails={() => setDetailsOpen(true)}
                />
              ) : (
                <DiscoveryComplete
                  key="discovery-complete"
                  total={total}
                  stats={stats}
                  profile={batch.response.profile}
                  model={batch.response.recommendation_info.model}
                  onImportAnother={handleImportAnother}
                  onRequestMore={onRequestMore}
                />
              )}
            </AnimatePresence>
          </div>

          <div className="flex min-h-[80px] shrink-0 flex-col items-center justify-center gap-3 sm:min-h-[90px]">
            {error ? (
              <p
                role="alert"
                className="max-w-md rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-2.5 text-center text-xs leading-relaxed text-danger/90"
              >
                {error}
              </p>
            ) : null}
            {current ? (
              <SwipeControls
                canUndo={decisions.length > 0}
                disabled={busy}
                onUndo={handleUndo}
                onPass={() => handleDecision("passed")}
                onSeen={() => handleDecision("seen")}
                onInterested={() => handleDecision("interested")}
              />
            ) : null}
          </div>

          <footer className="pb-1 pt-1 text-center">
            <TmdbAttribution />
          </footer>
        </div>
      </div>

      <MovieDetails
        open={detailsOpen}
        recommendation={current}
        model={batch.response.recommendation_info.model}
        onClose={() => setDetailsOpen(false)}
      />

      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>
    </main>
  );
}
