"use client";

import { Star } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { FilmDetailBody } from "@/components/films/FilmDetailBody";
import { Dialog } from "@/components/ui/Dialog";
import { fetchFilmDetail } from "@/lib/films";
import type { FilmDetail } from "@/types/films";
import type { Recommendation } from "@/types/recommendations";

interface MovieDetailsProps {
  open: boolean;
  recommendation: Recommendation | null;
  model: string;
  onClose: () => void;
}

function fallbackDetail(recommendation: Recommendation): FilmDetail {
  return {
    internal_movie_id: recommendation.internal_movie_id,
    movie_id: String(recommendation.movie_id),
    title: recommendation.title,
    year: recommendation.year,
    release_date: null,
    genres: recommendation.genres,
    tmdb_id: recommendation.tmdb_id,
    poster_path: recommendation.poster_path,
    backdrop_path: recommendation.backdrop_path,
    ratings_count: 0,
    rating_mean: 0,
    weighted_rating: recommendation.score,
    user_rating: null,
    user_list_status: null,
    summary: null,
  };
}

function ScorePanel({
  recommendation,
  model,
}: {
  recommendation: Recommendation;
  model: string;
}) {
  return (
    <div className="mt-6 flex items-center justify-between gap-4 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3.5">
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
      <p className="max-w-[10rem] text-right text-[10px] leading-relaxed text-white/45">
        Estimated from your FilmFinder ratings.
      </p>
    </div>
  );
}

function MovieDetailsContent({
  recommendation,
  model,
  titleId,
}: {
  recommendation: Recommendation;
  model: string;
  titleId: string;
}) {
  const [detail, setDetail] = useState<FilmDetail>(() =>
    fallbackDetail(recommendation),
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchFilmDetail(recommendation.internal_movie_id, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) {
          setDetail(payload);
          setError(null);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setError(
            "The extended film details are unavailable right now.",
          );
        }
      });
    return () => controller.abort();
  }, [recommendation.internal_movie_id]);

  return (
    <FilmDetailBody
      film={detail}
      titleId={titleId}
      isAuthenticated={false}
      readOnlyActions
      error={error}
      extra={
        <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
          Pick #{recommendation.rank}
        </p>
      }
      footer={
        <ScorePanel recommendation={recommendation} model={model} />
      }
    />
  );
}

export function MovieDetails({
  open,
  recommendation,
  model,
  onClose,
}: MovieDetailsProps) {
  const titleId = useId();
  return (
    <Dialog
      open={open}
      onClose={onClose}
      titleId={titleId}
      className="max-w-2xl"
    >
      {open && recommendation ? (
        <MovieDetailsContent
          key={recommendation.internal_movie_id}
          recommendation={recommendation}
          model={model}
          titleId={titleId}
        />
      ) : null}
    </Dialog>
  );
}
