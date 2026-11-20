"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import type { Recommendation } from "@/types/recommendations";

interface MovieBackdropProps {
  recommendation: Recommendation | null;
  priority?: boolean;
}

function BackdropFallback() {
  return (
    <div
      aria-hidden="true"
      className="h-full w-full bg-[radial-gradient(120%_90%_at_70%_5%,#242424_0%,#101010_42%,#080808_82%)]"
    />
  );
}

export function MovieBackdrop({
  recommendation,
  priority = false,
}: MovieBackdropProps) {
  const reduceMotion = useReducedMotion();
  const key = recommendation?.backdrop_path ?? "fallback";

  return (
    <div aria-hidden="true" className="absolute inset-0 overflow-hidden">
      <AnimatePresence initial={false}>
        <motion.div
          key={key}
          className="absolute inset-0"
          initial={reduceMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduceMotion ? 0 : 0.9, ease: "easeInOut" }}
        >
          <TmdbMedia
            path={recommendation?.backdrop_path}
            variant="backdrop"
            alt=""
            sizes="100vw"
            preload={priority}
            imageClassName="scale-[1.04] blur-[1px]"
            fallback={<BackdropFallback />}
          />
        </motion.div>
      </AnimatePresence>
      <div className="absolute inset-0 bg-black/55" />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(8,8,8,0.82)_0%,rgba(8,8,8,0.28)_32%,rgba(8,8,8,0.55)_68%,#080808_100%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(90%_70%_at_50%_42%,transparent_0%,rgba(8,8,8,0.35)_62%,rgba(8,8,8,0.9)_100%)]" />
    </div>
  );
}
