"use client";

import { ExternalLink, Film, Star } from "lucide-react";
import { useId } from "react";
import { Dialog } from "@/components/ui/Dialog";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import { getTmdbMovieUrl } from "@/lib/tmdb";
import type { Recommendation } from "@/types/recommendations";

interface MovieDetailsProps {
  open: boolean;
  recommendation: Recommendation | null;
  model: string;
  onClose: () => void;
}

function DetailsPosterFallback({
  title,
  year,
}: {
  title: string;
  year: number | null;
}) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 bg-[radial-gradient(120%_100%_at_30%_0%,#242424,#101010_55%,#0a0a0a)] p-3 text-center">
      <Film aria-hidden="true" className="h-6 w-6 text-white/25" />
      <p className="text-[11px] leading-tight text-white/60">{title}</p>
      {year ? <p className="text-[10px] text-white/45">{year}</p> : null}
    </div>
  );
}

export function MovieDetails({
  open,
  recommendation,
  model,
  onClose,
}: MovieDetailsProps) {
  const titleId = useId();
  const tmdbUrl = getTmdbMovieUrl(recommendation?.tmdb_id);

  return (
    <Dialog open={open} onClose={onClose} titleId={titleId} className="max-w-md">
      {recommendation ? (
        <div className="p-7 pt-16 sm:p-8 sm:pt-16">
          <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
            Pick #{recommendation.rank}
          </p>
          <div className="mt-5 flex gap-5">
            <div className="h-40 w-[6.75rem] shrink-0 overflow-hidden rounded-xl border border-white/10 bg-surface">
              <TmdbMedia
                path={recommendation.poster_path}
                variant="poster"
                alt={`${recommendation.title} poster`}
                sizes="108px"
                fallback={
                  <DetailsPosterFallback
                    title={recommendation.title}
                    year={recommendation.year}
                  />
                }
              />
            </div>
            <div className="min-w-0 flex-1">
              <h2
                id={titleId}
                className="text-2xl font-semibold leading-tight tracking-[-0.03em] text-white"
              >
                {recommendation.title}
              </h2>
              <p className="mt-2 text-sm text-white/55">
                {recommendation.year ?? "Release year unavailable"}
              </p>
              {recommendation.genres.length > 0 ? (
                <ul className="mt-4 flex flex-wrap gap-1.5" aria-label="Genres">
                  {recommendation.genres.map((genre) => (
                    <li
                      key={genre}
                      className="rounded-full border border-white/10 px-2.5 py-1 text-[10px] text-white/55"
                    >
                      {genre}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </div>

          <div className="mt-7 flex items-center justify-between gap-4 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3.5">
            <div>
              <p className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-white/50">
                <Star
                  aria-hidden="true"
                  className="h-3 w-3 fill-white/70 text-white/70"
                />
                {model} predicted rating
              </p>
              <p className="mt-1 text-2xl font-semibold tracking-tight text-white">
                {recommendation.score.toFixed(1)}
              </p>
            </div>
            <p className="max-w-[9rem] text-right text-[10px] leading-relaxed text-white/45">
              Estimated from the films you rated on Letterboxd.
            </p>
          </div>

          <div className="mt-6 border-t border-white/10 pt-5">
            {tmdbUrl ? (
              <a
                href={tmdbUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex h-10 items-center gap-2 rounded-full border border-white/15 px-4 text-xs font-medium text-white/70 transition-colors hover:border-white/30 hover:bg-white/5 hover:text-white"
              >
                View on TMDB
                <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
              </a>
            ) : (
              <p className="text-xs text-white/45">
                A TMDB page is not available for this film.
              </p>
            )}
          </div>
        </div>
      ) : null}
    </Dialog>
  );
}
