"use client";

import {
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
  type PanInfo,
} from "motion/react";
import { Film, Info, Star } from "lucide-react";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import type { ExitDirection } from "@/hooks/useDiscover";
import type { DecisionKind, Recommendation } from "@/types/recommendations";

interface MovieCardProps {
  recommendation: Recommendation;
  exitDirection: ExitDirection;
  onDecision: (kind: DecisionKind) => void;
  onOpenDetails: () => void;
}

function PosterFallback({
  title,
  year,
}: {
  title: string;
  year: number | null;
}) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-[radial-gradient(120%_100%_at_30%_0%,#242424,#101010_55%,#0a0a0a)] p-6 text-center">
      <Film aria-hidden="true" className="h-8 w-8 text-white/25" />
      <div>
        <p className="text-balance text-sm font-medium text-white/70">{title}</p>
        {year ? <p className="mt-1 text-xs text-white/50">{year}</p> : null}
      </div>
    </div>
  );
}

const cardVariants = {
  initial: { opacity: 0, y: 18, scale: 0.985 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: (direction: ExitDirection) => {
    if (direction === "right") {
      return { x: 640, rotate: 12, opacity: 0 };
    }
    if (direction === "left") {
      return { x: -640, rotate: -12, opacity: 0 };
    }
    return { y: 160, opacity: 0, scale: 0.96 };
  },
};

export function MovieCard({
  recommendation,
  exitDirection,
  onDecision,
  onOpenDetails,
}: MovieCardProps) {
  const reduceMotion = useReducedMotion();
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-320, 0, 320], [-8, 0, 8]);
  const interestedOpacity = useTransform(x, [30, 140], [0, 1]);
  const passedOpacity = useTransform(x, [-140, -30], [1, 0]);

  const handleDragEnd = (_event: unknown, info: PanInfo) => {
    const offset = info.offset.x;
    const velocity = info.velocity.x;
    if (offset > 96 || velocity > 640) {
      onDecision("interested");
      return;
    }
    if (offset < -96 || velocity < -640) {
      onDecision("passed");
    }
  };

  const genreLabel = recommendation.genres.slice(0, 3).join(" · ");

  return (
    <motion.div
      className="flex w-[min(78vw,calc((100svh-370px)*0.6667),340px)] cursor-grab touch-pan-y select-none flex-col active:cursor-grabbing"
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.18}
      dragSnapToOrigin
      style={{ x, rotate }}
      onDragEnd={handleDragEnd}
      custom={exitDirection}
      variants={cardVariants}
      initial={reduceMotion ? false : "initial"}
      animate="animate"
      exit={reduceMotion ? { opacity: 0 } : "exit"}
      transition={{
        duration: reduceMotion ? 0 : 0.32,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <div className="relative aspect-[2/3] w-full overflow-hidden rounded-2xl border border-white/10 bg-surface shadow-[0_35px_90px_-30px_rgba(0,0,0,0.95)]">
        <TmdbMedia
          path={recommendation.poster_path}
          variant="poster"
          alt={`${recommendation.title}${recommendation.year ? ` (${recommendation.year})` : ""} poster`}
          sizes="(max-width: 640px) 78vw, 340px"
          fallback={
            <PosterFallback
              title={recommendation.title}
              year={recommendation.year}
            />
          }
        />

        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,transparent_55%,rgba(0,0,0,0.55)_100%)]"
        />

        <motion.span
          aria-hidden="true"
          style={{ opacity: interestedOpacity }}
          className="pointer-events-none absolute left-4 top-4 -rotate-6 rounded-md border-2 border-white px-2.5 py-1 text-[10px] font-bold tracking-[0.18em] text-white"
        >
          INTERESTED
        </motion.span>
        <motion.span
          aria-hidden="true"
          style={{ opacity: passedOpacity }}
          className="pointer-events-none absolute right-4 top-4 rotate-6 rounded-md border-2 border-white/70 px-2.5 py-1 text-[10px] font-bold tracking-[0.18em] text-white/80"
        >
          NOT FOR ME
        </motion.span>

        <span className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-2.5 py-1.5 text-[11px] font-medium text-white backdrop-blur-md">
          <Star
            aria-hidden="true"
            className="h-3 w-3 fill-white text-white"
          />
          {recommendation.score.toFixed(1)}
          <span className="sr-only"> predicted rating</span>
        </span>
        <button
          type="button"
          onClick={onOpenDetails}
          onPointerDown={(event) => event.stopPropagation()}
          aria-haspopup="dialog"
          aria-label={`View details for ${recommendation.title}`}
          className="absolute bottom-3 right-3 inline-flex h-8 items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-3 text-[11px] font-medium text-white/70 backdrop-blur-md transition-colors hover:bg-white/15 hover:text-white"
        >
          <Info aria-hidden="true" className="h-3.5 w-3.5" />
          Details
        </button>
      </div>

      <div className="w-full pt-3.5 text-center">
        <h2 className="line-clamp-3 text-balance text-lg font-semibold leading-tight tracking-[-0.025em] text-white sm:text-xl">
          {recommendation.title}
        </h2>
        <p className="mt-1.5 text-xs text-white/55">
          {[recommendation.year, genreLabel].filter(Boolean).join(" · ")}
        </p>
      </div>
    </motion.div>
  );
}
