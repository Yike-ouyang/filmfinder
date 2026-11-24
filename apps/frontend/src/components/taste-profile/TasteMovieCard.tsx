"use client";

import { Eye, Film, SkipForward } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { RatingStars } from "@/components/films/RatingStars";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import type { TasteProfileMovie } from "@/types/tasteProfile";

interface TasteMovieCardProps {
  movie: TasteProfileMovie;
  rating: number | null;
  onRate: (rating: number) => void;
  onSkip: () => void;
  onViewDetails?: () => void;
  disabled?: boolean;
}

function PosterFallback({ title, year }: { title: string; year: number | null }) {
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

export function TasteMovieCard({
  movie,
  rating,
  onRate,
  onSkip,
  onViewDetails,
  disabled = false,
}: TasteMovieCardProps) {
  const reduceMotion = useReducedMotion();
  const genreLabel = movie.genres.slice(0, 3).join(" · ");
  const existing = movie.user_rating !== null && movie.user_rating === rating;

  return (
    <motion.div
      className="flex w-[min(86vw,380px)] flex-col"
      initial={reduceMotion ? false : { opacity: 0, y: 18, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -14, scale: 0.985 }}
      transition={{
        duration: reduceMotion ? 0 : 0.3,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <div className="relative aspect-[2/3] w-full overflow-hidden rounded-2xl border border-white/10 bg-surface shadow-[0_35px_90px_-30px_rgba(0,0,0,0.95)]">
        <TmdbMedia
          path={movie.poster_path}
          variant="poster"
          alt={`${movie.title}${movie.year ? ` (${movie.year})` : ""} poster`}
          sizes="(max-width: 640px) 86vw, 380px"
          fallback={<PosterFallback title={movie.title} year={movie.year} />}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,transparent_58%,rgba(0,0,0,0.5)_100%)]"
        />
      </div>

      <div className="w-full pt-3.5 text-center">
        <h2 className="line-clamp-2 text-balance text-lg font-semibold leading-tight tracking-[-0.025em] text-white sm:text-xl">
          {movie.title}
        </h2>
        <p className="mt-1.5 text-xs text-white/55">
          {[movie.year, genreLabel].filter(Boolean).join(" · ")}
        </p>
      </div>

      <div className="mt-5 flex w-full flex-col items-center gap-3">
        <div className="flex min-h-[34px] items-center gap-3">
          <RatingStars
            value={rating}
            onChange={onRate}
            disabled={disabled}
            size="lg"
            label={`Rate ${movie.title}`}
          />
          <span className="min-w-[42px] text-left text-xs text-white/55">
            {rating !== null ? `${rating}/10` : "Not rated"}
          </span>
        </div>
        {existing ? (
          <p className="text-[10px] uppercase tracking-[0.14em] text-white/35">
            Your library rating
          </p>
        ) : null}
        <div className="flex w-full flex-wrap items-center justify-center gap-2">
          <button
            type="button"
            onClick={onSkip}
            disabled={disabled}
            className="inline-flex h-10 items-center gap-1.5 rounded-full border border-white/15 bg-black/30 px-4 text-xs font-medium text-white/70 backdrop-blur-md transition-colors hover:border-white/30 hover:bg-white/10 hover:text-white disabled:pointer-events-none disabled:opacity-40"
          >
            <SkipForward aria-hidden="true" className="h-3.5 w-3.5" />
            Haven&apos;t Seen
          </button>
          {onViewDetails ? (
            <button
              type="button"
              onClick={onViewDetails}
              className="inline-flex h-10 items-center gap-1.5 rounded-full border border-white/10 px-4 text-xs font-medium text-white/55 transition-colors hover:border-white/20 hover:bg-white/5 hover:text-white"
            >
              <Eye aria-hidden="true" className="h-3.5 w-3.5" />
              View details
            </button>
          ) : null}
        </div>
      </div>
    </motion.div>
  );
}
