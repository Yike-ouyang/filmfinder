"use client";

import { Film } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { TasteActions } from "@/components/taste-profile/TasteActions";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import type {
  TasteProfileAction,
  TasteProfileMovie,
} from "@/types/tasteProfile";

interface TasteMovieCardProps {
  movie: TasteProfileMovie;
  onAction: (action: TasteProfileAction) => void;
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
  onAction,
  disabled = false,
}: TasteMovieCardProps) {
  const reduceMotion = useReducedMotion();
  const genreLabel = movie.genres.slice(0, 3).join(" · ");

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

      <div className="mt-4 w-full">
        <TasteActions onAction={onAction} disabled={disabled} />
      </div>
    </motion.div>
  );
}
