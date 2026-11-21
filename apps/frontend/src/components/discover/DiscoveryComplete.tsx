"use client";

import { motion, useReducedMotion } from "motion/react";
import { LoaderCircle, RefreshCw, Sparkles } from "lucide-react";
import { useState } from "react";
import type {
  DecisionKind,
  ProfileSummary,
  RequestMoreResult,
} from "@/types/recommendations";

interface DiscoveryCompleteProps {
  total: number;
  stats: Record<DecisionKind, number>;
  profile: ProfileSummary;
  model: string;
  onImportAnother: () => void;
  onRequestMore?: () => Promise<RequestMoreResult>;
}

export function DiscoveryComplete({
  total,
  stats,
  profile,
  model,
  onImportAnother,
  onRequestMore,
}: DiscoveryCompleteProps) {
  const reduceMotion = useReducedMotion();
  const [requesting, setRequesting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageKind, setMessageKind] = useState<"info" | "error">("info");

  const requestMore = async () => {
    if (!onRequestMore || requesting) {
      return;
    }
    setRequesting(true);
    setMessage(null);
    try {
      const result = await onRequestMore();
      if (result.status === "no_new") {
        setMessageKind("info");
        setMessage(
          "No new recommendations right now. Try again after updating your preferences or importing a fresh profile.",
        );
      } else if (result.status === "error") {
        setMessageKind("error");
        setMessage(
          result.message ??
            "New recommendations could not be prepared. Try again in a moment.",
        );
      }
    } catch {
      setMessageKind("error");
      setMessage("New recommendations could not be prepared. Try again.");
    } finally {
      setRequesting(false);
    }
  };

  return (
    <motion.section
      aria-labelledby="discovery-complete-title"
      className="w-full max-w-lg rounded-3xl border border-white/10 bg-black/45 px-6 py-8 text-center shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:px-10 sm:py-10"
      initial={reduceMotion ? false : { opacity: 0, y: 20, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0 }}
      transition={{
        duration: reduceMotion ? 0 : 0.45,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-white/10 bg-white/[0.04]">
        <Sparkles aria-hidden="true" className="h-5 w-5 text-white/70" />
      </span>
      <p className="mt-5 text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
        End of this set
      </p>
      <h2
        id="discovery-complete-title"
        className="mt-4 text-balance text-3xl font-semibold leading-tight tracking-[-0.035em] text-white sm:text-4xl"
      >
        You&apos;ve reached the end of these discoveries.
      </h2>
      <p className="mt-4 text-sm text-white/55">
        Keep going with a fresh set, or import an updated Letterboxd profile.
      </p>

      <dl className="mt-6 grid grid-cols-3 gap-2 rounded-2xl border border-white/10 bg-white/[0.02] px-2 py-4">
        <div>
          <dt className="text-[9px] uppercase tracking-[0.16em] text-white/55">
            Interested
          </dt>
          <dd className="mt-1.5 text-xl font-semibold text-white">
            {stats.interested}
          </dd>
        </div>
        <div className="border-x border-white/10">
          <dt className="text-[9px] uppercase tracking-[0.16em] text-white/55">
            Passed
          </dt>
          <dd className="mt-1.5 text-xl font-semibold text-white">
            {stats.passed}
          </dd>
        </div>
        <div>
          <dt className="text-[9px] uppercase tracking-[0.16em] text-white/55">
            Seen
          </dt>
          <dd className="mt-1.5 text-xl font-semibold text-white">
            {stats.seen}
          </dd>
        </div>
      </dl>

      <p className="mt-4 text-[11px] leading-relaxed text-white/55">
        {total} films explored · {profile.ratings_matched} of{" "}
        {profile.ratings_found} ratings matched · {model}
      </p>

      {message ? (
        <p
          role={messageKind === "error" ? "alert" : "status"}
          className={
            messageKind === "error"
              ? "mt-5 rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-3 text-xs leading-relaxed text-danger/90"
              : "mt-5 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-xs leading-relaxed text-white/55"
          }
        >
          {message}
        </p>
      ) : null}

      <div className="mt-6 flex flex-col gap-3">
        {onRequestMore ? (
          <button
            type="button"
            onClick={requestMore}
            disabled={requesting}
            className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90 disabled:cursor-wait disabled:opacity-70"
          >
            {requesting ? (
              <LoaderCircle
                aria-hidden="true"
                className="h-4 w-4 animate-spin"
              />
            ) : (
              <RefreshCw aria-hidden="true" className="h-4 w-4" />
            )}
            {requesting ? "Finding more films..." : "Get more recommendations"}
          </button>
        ) : null}
        <button
          type="button"
          onClick={onImportAnother}
          disabled={requesting}
          className={
            onRequestMore
              ? "inline-flex h-11 w-full items-center justify-center rounded-full border border-white/10 px-6 text-sm font-medium text-white/65 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-40"
              : "inline-flex h-11 w-full items-center justify-center rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90 disabled:opacity-40 sm:w-auto"
          }
        >
          Import another profile
        </button>
      </div>
    </motion.section>
  );
}
