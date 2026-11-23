"use client";

import { Film, Star } from "lucide-react";
import { RatingStars } from "@/components/films/RatingStars";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import type { FilmListItem } from "@/types/films";

interface FilmCardProps {
  film: FilmListItem;
  onOpen: (filmId: number) => void;
}

function PosterFallback({ title, year }: { title: string; year: number | null }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-[radial-gradient(120%_100%_at_30%_0%,#242424,#101010_55%,#0a0a0a)] p-3 text-center">
      <Film aria-hidden="true" className="h-6 w-6 text-white/25" />
      <p className="line-clamp-3 text-[11px] leading-tight text-white/60">
        {title}
      </p>
      {year ? <p className="text-[10px] text-white/40">{year}</p> : null}
    </div>
  );
}

export function FilmCard({ film, onOpen }: FilmCardProps) {
  const year = film.year ?? null;
  return (
    <button
      type="button"
      onClick={() => onOpen(film.internal_movie_id)}
      aria-label={`View ${film.title}${year ? ` (${year})` : ""}`}
      className="group w-full text-left focus-visible:outline-none"
    >
      <div className="relative aspect-[2/3] w-full overflow-hidden rounded-xl border border-white/10 bg-surface shadow-[0_18px_45px_-25px_rgba(0,0,0,0.9)] transition-[border-color,transform] duration-300 group-hover:-translate-y-0.5 group-hover:border-white/25 group-focus-visible:ring-2 group-focus-visible:ring-white/70">
        <TmdbMedia
          path={film.poster_path}
          variant="poster"
          alt={`${film.title} poster`}
          sizes="(max-width: 640px) 45vw, (max-width: 1024px) 22vw, 190px"
          fallback={<PosterFallback title={film.title} year={year} />}
          imageClassName="transition-transform duration-500 group-hover:scale-[1.03]"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,transparent_58%,rgba(0,0,0,0.42)_100%)]"
        />
      </div>

      <div className="mt-2.5 min-w-0">
        <h3 className="line-clamp-2 text-sm font-medium leading-tight text-white">
          {film.title}
        </h3>
        <p className="mt-1 text-[11px] text-white/45">
          {year ?? "Year unavailable"}
        </p>
        <div className="mt-1.5 flex min-h-[18px] items-center justify-between gap-2">
          {film.user_rating !== null ? (
            <span
              className="inline-flex items-center"
              aria-label={`Your rating: ${film.user_rating} out of 10`}
            >
              <RatingStars
                value={film.user_rating}
                readOnly
                size="sm"
                label="Your rating"
              />
            </span>
          ) : (
            <span className="text-[10px] uppercase tracking-[0.08em] text-white/30">
              Not rated
            </span>
          )}
          <span
            className="inline-flex items-center gap-1 text-[10px] text-white/40"
            aria-label={`Average rating: ${film.rating_mean.toFixed(1)} out of 10`}
          >
            <Star aria-hidden="true" className="h-3 w-3" />
            Avg {film.rating_mean.toFixed(1)}
          </span>
        </div>
      </div>
    </button>
  );
}
