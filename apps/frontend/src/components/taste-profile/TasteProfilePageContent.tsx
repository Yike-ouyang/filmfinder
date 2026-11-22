"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ArrowRight,
  LoaderCircle,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { TasteMovieCard } from "@/components/taste-profile/TasteMovieCard";
import { TasteProfileBackdrop } from "@/components/taste-profile/TasteProfileBackdrop";
import { TmdbAttribution } from "@/components/ui/TmdbAttribution";
import {
  ApiError,
  errorMessage,
  requestTasteProfileMovies,
  requestTasteProfileRecommendations,
} from "@/lib/api";
import {
  RECOMMENDATION_LIMIT,
  TASTE_PROFILE_LOADING_MESSAGES,
} from "@/lib/constants";
import { saveBatch } from "@/lib/storage";
import {
  countRatedAnswers,
  hasEnoughRatings,
  initialTasteProfileState,
  isTasteProfileComplete,
  ratingForAction,
  tasteProfileReducer,
} from "@/lib/tasteProfile";
import {
  clearTasteProfileDraft,
  loadTasteProfileDraft,
  saveTasteProfileDraft,
} from "@/lib/tasteProfileDraft";
import type {
  TasteProfileAction,
  TasteProfileAnswer,
  TasteProfileMoviesResponse,
} from "@/types/tasteProfile";

type LoadStatus = "loading" | "ready" | "error";
type SubmitStatus = "idle" | "loading" | "ready" | "error";

function CenteredShell({ children }: { children: React.ReactNode }) {
  return (
    <main
      id="main"
      className="relative flex min-h-svh items-center justify-center overflow-hidden bg-background px-5 py-24 sm:px-8"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_80%_at_50%_0%,#1b1b1b_0%,#0d0d0d_45%,#080808_80%)]"
      />
      <div className="relative z-10 w-full max-w-xl">{children}</div>
    </main>
  );
}

function LoadingPanel({ message }: { message: string }) {
  return (
    <CenteredShell>
      <div
        role="status"
        aria-live="polite"
        className="mx-auto flex max-w-md flex-col items-center gap-4 text-center text-white/60"
      >
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin text-white/60"
        />
        <p className="text-sm">{message}</p>
      </div>
    </CenteredShell>
  );
}

function ErrorPanel({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <CenteredShell>
      <div className="rounded-3xl border border-white/10 bg-black/45 px-6 py-8 text-center shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:px-10">
        <span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06]">
          <AlertCircle aria-hidden="true" className="h-5 w-5 text-danger" />
        </span>
        <h1 className="mt-5 text-2xl font-semibold tracking-[-0.03em] text-white">
          Something went wrong
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-white/55">{message}</p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-6 inline-flex h-11 items-center justify-center gap-2 rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90"
        >
          <RotateCcw aria-hidden="true" className="h-4 w-4" />
          Try again
        </button>
      </div>
    </CenteredShell>
  );
}

function IntroPanel({
  data,
  onStart,
  isAuthenticated,
}: {
  data: TasteProfileMoviesResponse;
  onStart: () => void;
  isAuthenticated: boolean;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.section
      aria-labelledby="taste-profile-title"
      className="rounded-3xl border border-white/10 bg-black/45 px-6 py-8 text-center shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:px-10 sm:py-10"
      initial={reduceMotion ? false : { opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: reduceMotion ? 0 : 0.4,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-white/10 bg-white/[0.04]">
        <Sparkles aria-hidden="true" className="h-5 w-5 text-white/70" />
      </span>
      <p className="mt-5 text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
        Quick Taste Profile
      </p>
      <h1
        id="taste-profile-title"
        className="mt-4 text-balance text-3xl font-semibold leading-tight tracking-[-0.035em] text-white sm:text-4xl"
      >
        Rate a few films. Find your next favorite.
      </h1>
      <p className="mx-auto mt-4 max-w-md text-sm leading-relaxed text-white/55">
        Go through {data.movies.length} films and tell us what you thought.
        We&apos;ll use your ratings to find movies made for your taste.
      </p>

      <dl className="mt-6 grid grid-cols-3 gap-2 rounded-2xl border border-white/10 bg-white/[0.02] px-2 py-4">
        <div>
          <dt className="text-[9px] uppercase tracking-[0.16em] text-white/55">
            Films
          </dt>
          <dd className="mt-1.5 text-xl font-semibold text-white">
            {data.movies.length}
          </dd>
        </div>
        <div className="border-x border-white/10">
          <dt className="text-[9px] uppercase tracking-[0.16em] text-white/55">
            Minimum
          </dt>
          <dd className="mt-1.5 text-xl font-semibold text-white">
            {data.min_ratings}
          </dd>
        </div>
        <div>
          <dt className="text-[9px] uppercase tracking-[0.16em] text-white/55">
            Time
          </dt>
          <dd className="mt-1.5 text-xl font-semibold text-white">~2 min</dd>
        </div>
      </dl>

      <button
        type="button"
        onClick={onStart}
        className="group mt-6 inline-flex h-11 items-center justify-center gap-2 rounded-full bg-white px-7 text-sm font-medium text-black transition-opacity hover:opacity-90"
      >
        Start rating
        <ArrowRight
          aria-hidden="true"
          className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5"
        />
      </button>

      <p className="mt-5 text-[11px] leading-relaxed text-white/40">
        {isAuthenticated
          ? "Your ratings will be saved to your account."
          : "No account needed. Want to keep your ratings? "}
        {!isAuthenticated ? (
          <Link
            href="/signup"
            className="text-white/70 underline decoration-white/20 underline-offset-2 hover:text-white"
          >
            Create an account
          </Link>
        ) : null}
      </p>
    </motion.section>
  );
}

function DraftPanel({
  isAuthenticated,
  saving,
  error,
  onSave,
  onDiscard,
}: {
  isAuthenticated: boolean;
  saving: boolean;
  error: string | null;
  onSave: () => void;
  onDiscard: () => void;
}) {
  return (
    <div className="mt-4 rounded-2xl border border-white/10 bg-black/45 px-5 py-4 text-center backdrop-blur-xl">
      <p className="text-sm font-medium text-white">
        {isAuthenticated
          ? "Save your Taste Profile to your account"
          : "You have an unsaved Taste Profile in this browser"}
      </p>
      <p className="mt-1 text-[11px] leading-relaxed text-white/50">
        {isAuthenticated
          ? "Keep these ratings so they can be used again next time."
          : "Create an account or log in to keep these ratings."}
      </p>
      {error ? (
        <p className="mt-2 text-[11px] text-danger">{error}</p>
      ) : null}
      <div className="mt-3 flex flex-col justify-center gap-2 sm:flex-row">
        {isAuthenticated ? (
          <button
            type="button"
            onClick={onSave}
            disabled={saving}
            className="inline-flex h-9 items-center justify-center rounded-full bg-white px-4 text-xs font-medium text-black transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save my profile"}
          </button>
        ) : (
          <>
            <Link
              href="/signup"
              className="inline-flex h-9 items-center justify-center rounded-full bg-white px-4 text-xs font-medium text-black transition-opacity hover:opacity-90"
            >
              Create an account
            </Link>
            <Link
              href="/login"
              className="inline-flex h-9 items-center justify-center rounded-full border border-white/10 px-4 text-xs font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
            >
              Log in
            </Link>
          </>
        )}
        <button
          type="button"
          onClick={onDiscard}
          className="inline-flex h-9 items-center justify-center rounded-full px-3 text-xs font-medium text-white/40 transition-colors hover:text-white/70"
        >
          Discard
        </button>
      </div>
    </div>
  );
}

function QuizHeader({
  current,
  total,
  canUndo,
  onUndo,
}: {
  current: number;
  total: number;
  canUndo: boolean;
  onUndo: () => void;
}) {
  const label = `${current} / ${total} movies`;
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="min-w-0 flex-1">
        <p
          className="font-mono text-[11px] tracking-[0.16em] text-white/50"
          aria-label={label}
        >
          <span className="text-white/80">{current}</span>
          <span className="mx-1 text-white/20">/</span>
          {total} movies
        </p>
        <div
          role="progressbar"
          aria-valuemin={1}
          aria-valuemax={total}
          aria-valuenow={current}
          aria-label="Taste Profile progress"
          className="mt-2 h-1 w-full max-w-xs overflow-hidden rounded-full bg-white/10"
        >
          <div
            className="h-full rounded-full bg-white transition-[width] duration-300 ease-out"
            style={{ width: `${(current / total) * 100}%` }}
          />
        </div>
      </div>
      <button
        type="button"
        onClick={onUndo}
        disabled={!canUndo}
        aria-label="Undo the previous rating (Backspace)"
        title="Undo (Backspace)"
        className="inline-flex h-9 items-center gap-1.5 rounded-full border border-white/10 bg-black/35 px-3 text-[11px] font-medium text-white/60 backdrop-blur-md transition-colors hover:border-white/20 hover:text-white disabled:pointer-events-none disabled:opacity-30"
      >
        <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
        Undo
      </button>
    </div>
  );
}

function FinalPanel({
  rated,
  minimum,
  onReview,
  onRestart,
}: {
  rated: number;
  minimum: number;
  onReview: () => void;
  onRestart: () => void;
}) {
  return (
    <CenteredShell>
      <section
        aria-labelledby="taste-profile-final-title"
        className="rounded-3xl border border-white/10 bg-black/45 px-6 py-8 text-center shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:px-10"
      >
        <span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-white/10 bg-white/[0.04]">
          <Sparkles aria-hidden="true" className="h-5 w-5 text-white/70" />
        </span>
        <h1
          id="taste-profile-final-title"
          className="mt-5 text-balance text-2xl font-semibold tracking-[-0.03em] text-white sm:text-3xl"
        >
          A few more ratings would help.
        </h1>
        <p className="mx-auto mt-3 max-w-sm text-sm leading-relaxed text-white/55">
          You rated {rated} of {minimum} required movies. Rate at least{" "}
          {minimum} films so we can build a useful profile for you.
        </p>
        <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
          <button
            type="button"
            onClick={onReview}
            className="inline-flex h-11 items-center justify-center rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90"
          >
            Review my answers
          </button>
          <button
            type="button"
            onClick={onRestart}
            className="inline-flex h-11 items-center justify-center rounded-full border border-white/10 px-6 text-sm font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
          >
            Start over
          </button>
        </div>
      </section>
    </CenteredShell>
  );
}

function SubmittingPanel({
  message,
  ready,
}: {
  message: string;
  ready: boolean;
}) {
  return (
    <CenteredShell>
      <div
        role="status"
        aria-live="polite"
        className="mx-auto flex max-w-md flex-col items-center gap-4 text-center"
      >
        <LoaderCircle
          aria-hidden="true"
          className="h-6 w-6 animate-spin text-white/70"
        />
        <p className="text-sm text-white/60">
          {ready ? "Opening your discoveries..." : message}
        </p>
      </div>
    </CenteredShell>
  );
}

export function TasteProfilePageContent() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();
  const [data, setData] = useState<TasteProfileMoviesResponse | null>(null);
  const [loadStatus, setLoadStatus] = useState<LoadStatus>("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [state, dispatch] = useReducer(
    tasteProfileReducer,
    initialTasteProfileState,
  );
  const [submitStatus, setSubmitStatus] = useState<SubmitStatus>("idle");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [messageIndex, setMessageIndex] = useState(0);
  const [announcement, setAnnouncement] = useState("");
  const [draftAnswers, setDraftAnswers] =
    useState<TasteProfileAnswer[] | null>(null);
  const [draftStatus, setDraftStatus] =
    useState<"idle" | "saving" | "error">("idle");
  const [draftError, setDraftError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const draftAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    router.prefetch("/discover");
  }, [router]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    requestTasteProfileMovies({ signal: controller.signal })
      .then((payload) => {
        if (!active) {
          return;
        }
        setData(payload);
        setLoadStatus("ready");
      })
      .catch((error) => {
        if (!active || controller.signal.aborted) {
          return;
        }
        if (error instanceof ApiError && error.code === "cancelled") {
          return;
        }
        setLoadError(errorMessage(error));
        setLoadStatus("error");
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [loadAttempt]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDraftAnswers(loadTasteProfileDraft());
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    return () => draftAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (submitStatus !== "loading") {
      return;
    }
    const interval = window.setInterval(() => {
      setMessageIndex((index) => index + 1);
    }, 2400);
    return () => window.clearInterval(interval);
  }, [submitStatus]);

  const total = data?.movies.length ?? 0;
  const complete = data
    ? isTasteProfileComplete(state, total)
    : false;
  const rated = countRatedAnswers(state.answers);
  const enough = data ? hasEnoughRatings(state.answers, data.min_ratings) : false;
  const current = data ? data.movies[state.index] ?? null : null;

  const submit = useCallback(async () => {
    if (!data || state.answers.length !== data.movies.length) {
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    setSubmitStatus("loading");
    setSubmitError(null);
    setMessageIndex(0);
    try {
      const result = await requestTasteProfileRecommendations({
        answers: state.answers,
        limit: RECOMMENDATION_LIMIT,
        signal: controller.signal,
      });
      if (controller.signal.aborted) {
        return;
      }
      if (!saveBatch(result, [])) {
        setSubmitError(
          "Your browser could not store this session. Allow session storage for this site and try again.",
        );
        setSubmitStatus("error");
        return;
      }
      if (!user) {
        saveTasteProfileDraft(state.answers);
      }
      setSubmitStatus("ready");
      router.push("/discover");
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      if (error instanceof ApiError && error.code === "cancelled") {
        return;
      }
      setSubmitError(errorMessage(error));
      setSubmitStatus("error");
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  }, [data, router, state.answers, user]);

  const saveDraft = useCallback(async () => {
    if (!draftAnswers) {
      return;
    }
    const controller = new AbortController();
    draftAbortRef.current = controller;
    setDraftStatus("saving");
    setDraftError(null);
    try {
      const result = await requestTasteProfileRecommendations({
        answers: draftAnswers,
        limit: RECOMMENDATION_LIMIT,
        signal: controller.signal,
      });
      if (controller.signal.aborted) {
        return;
      }
      if (!saveBatch(result, [])) {
        setDraftError(
          "Your browser could not store this session. Allow session storage for this site and try again.",
        );
        setDraftStatus("error");
        return;
      }
      clearTasteProfileDraft();
      setDraftAnswers(null);
      setDraftStatus("idle");
      router.push("/discover");
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      setDraftError(errorMessage(error));
      setDraftStatus("error");
    } finally {
      if (draftAbortRef.current === controller) {
        draftAbortRef.current = null;
      }
    }
  }, [draftAnswers, router]);

  const discardDraft = useCallback(() => {
    clearTasteProfileDraft();
    setDraftAnswers(null);
    setDraftStatus("idle");
    setDraftError(null);
  }, []);

  useEffect(() => {
    if (
      loadStatus !== "ready" ||
      !complete ||
      !enough ||
      submitStatus !== "idle"
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      void submit();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [complete, enough, loadStatus, submit, submitStatus]);

  const handleAction = useCallback(
    (action: TasteProfileAction) => {
      if (!current || submitStatus === "loading" || submitStatus === "ready") {
        return;
      }
      const rating = ratingForAction(action);
      dispatch({ type: "answer", movie: current, rating });
      const next = state.index + 1;
      if (rating === null) {
        setAnnouncement(
          `Skipped ${current.title} without a rating. ${next} of ${total} movies.`,
        );
      } else {
        setAnnouncement(
          `Rated ${current.title} ${rating} out of 10. ${next} of ${total} movies.`,
        );
      }
    },
    [current, state.index, submitStatus, total],
  );

  const handleUndo = useCallback(() => {
    if (state.index === 0) {
      return;
    }
    dispatch({ type: "undo" });
    setSubmitStatus("idle");
    setSubmitError(null);
    setAnnouncement("Restored the previous movie.");
  }, [state.index]);

  const handleRestart = useCallback(() => {
    dispatch({ type: "restart" });
    setSubmitStatus("idle");
    setSubmitError(null);
    setAnnouncement("Taste Profile restarted.");
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        !state.started ||
        complete ||
        submitStatus === "loading" ||
        submitStatus === "ready" ||
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
      if (event.key === "Backspace") {
        event.preventDefault();
        handleUndo();
        return;
      }
      const definition = ["1", "2", "3", "4"].includes(event.key)
        ? (["loved", "okay", "disliked", "no-opinion"] as const)[
            Number(event.key) - 1
          ]
        : null;
      if (definition) {
        event.preventDefault();
        handleAction(definition);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    complete,
    handleAction,
    handleUndo,
    state.started,
    submitStatus,
  ]);

  if (loadStatus === "loading") {
    return <LoadingPanel message="Loading your Taste Profile..." />;
  }

  if (loadStatus === "error" || !data) {
    return (
      <ErrorPanel
        message={loadError ?? "The Taste Profile could not be loaded."}
        onRetry={() => {
          setLoadStatus("loading");
          setLoadError(null);
          setLoadAttempt((attempt) => attempt + 1);
        }}
      />
    );
  }

  if (!state.started) {
    return (
      <main
        id="main"
        className="relative flex min-h-svh items-center justify-center overflow-hidden bg-background px-5 py-32 sm:px-8"
      >
        <TasteProfileBackdrop movie={data.movies[0] ?? null} priority />
        <div className="relative z-10 w-full max-w-xl">
          <IntroPanel
            data={data}
            isAuthenticated={Boolean(user)}
            onStart={() => {
              clearTasteProfileDraft();
              setDraftAnswers(null);
              dispatch({ type: "start" });
            }}
          />
          {draftAnswers && !authLoading ? (
            <DraftPanel
              isAuthenticated={Boolean(user)}
              saving={draftStatus === "saving"}
              error={draftError}
              onSave={() => void saveDraft()}
              onDiscard={discardDraft}
            />
          ) : null}
          <div className="mt-6 text-center">
            <TmdbAttribution />
          </div>
        </div>
      </main>
    );
  }

  if (complete) {
    if (submitStatus === "error") {
      return (
        <ErrorPanel
          message={submitError ?? "Your recommendations could not be prepared."}
          onRetry={() => void submit()}
        />
      );
    }
    if (!enough) {
      return (
        <FinalPanel
          rated={rated}
          minimum={data.min_ratings}
          onReview={handleUndo}
          onRestart={handleRestart}
        />
      );
    }
    return (
      <SubmittingPanel
        ready={submitStatus === "ready"}
        message={
          TASTE_PROFILE_LOADING_MESSAGES[
            messageIndex % TASTE_PROFILE_LOADING_MESSAGES.length
          ]
        }
      />
    );
  }

  return (
    <main
      id="main"
      className="relative flex min-h-svh flex-col overflow-hidden bg-background"
    >
      <TasteProfileBackdrop movie={current} priority={state.index === 0} />

      <h1 className="sr-only">Quick Taste Profile</h1>

      <div className="safe-bottom-tight relative z-10 flex min-h-svh flex-col px-5 pt-20 sm:px-8 sm:pt-24">
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col">
          <QuizHeader
            current={Math.min(state.index + 1, total)}
            total={total}
            canUndo={state.index > 0}
            onUndo={handleUndo}
          />

          <div className="flex flex-1 items-center justify-center py-4">
            <AnimatePresence mode="wait">
              {current ? (
                <TasteMovieCard
                  key={`${current.internal_movie_id}-${state.index}`}
                  movie={current}
                  onAction={handleAction}
                  disabled={
                    submitStatus === "loading" || submitStatus === "ready"
                  }
                />
              ) : null}
            </AnimatePresence>
          </div>

          <div className="flex min-h-[54px] shrink-0 flex-col items-center justify-end gap-2 pb-2">
            <p className="text-center text-[11px] leading-relaxed text-white/40">
              No opinion skips the movie without rating it. Use keys 1–4 and
              Backspace to undo.
            </p>
            <p className="text-center text-[10px] text-white/25">
              {!authLoading && user
                ? "Your ratings will be saved to your account."
                : "No account needed. Your answers stay in this browser."}
            </p>
          </div>

          <footer className="pb-1 pt-1 text-center">
            <TmdbAttribution />
          </footer>
        </div>
      </div>

      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>
    </main>
  );
}
