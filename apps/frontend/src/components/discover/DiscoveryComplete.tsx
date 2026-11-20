"use client";

import { motion, useReducedMotion } from "motion/react";
import { Sparkles } from "lucide-react";
import type { DecisionKind, ProfileSummary } from "@/types/recommendations";

interface DiscoveryCompleteProps {
  total: number;
  stats: Record<DecisionKind, number>;
  profile: ProfileSummary;
  model: string;
  onImportAnother: () => void;
}

export function DiscoveryComplete({
  total,
  stats,
  profile,
  model,
  onImportAnother,
}: DiscoveryCompleteProps) {
  const reduceMotion = useReducedMotion();

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
        Session complete
      </p>
      <h2
        id="discovery-complete-title"
        className="mt-4 text-balance text-3xl font-semibold leading-tight tracking-[-0.035em] text-white sm:text-4xl"
      >
        You&apos;ve reached the end of your discoveries.
      </h2>
      <p className="mt-4 text-sm text-white/55">
        Ready for another cinematic journey?
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

      <button
        type="button"
        onClick={onImportAnother}
        className="mt-6 inline-flex h-11 w-full items-center justify-center rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90 sm:w-auto"
      >
        Import another profile
      </button>
    </motion.section>
  );
}
