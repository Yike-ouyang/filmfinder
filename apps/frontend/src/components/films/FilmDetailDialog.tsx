"use client";

import { LoaderCircle, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useId, useState } from "react";
import { FilmDetailBody } from "@/components/films/FilmDetailBody";
import { Dialog } from "@/components/ui/Dialog";
import { errorMessage } from "@/lib/api";
import { fetchFilmDetail } from "@/lib/films";
import {
  removeFilmListStatus,
  saveFilmListStatus,
  type EditableFilmListStatus,
} from "@/lib/filmLists";
import { deleteFilmRating, setFilmRating } from "@/lib/ratings";
import type { FilmDetail, FilmListItem } from "@/types/films";

interface FilmDetailDialogProps {
  open: boolean;
  filmId: number | null;
  initialFilm?: FilmListItem | null;
  isAuthenticated: boolean;
  authNextPath?: string;
  readOnlyActions?: boolean;
  onClose: () => void;
  onFilmPatch?: (filmId: number, patch: Partial<FilmListItem>) => void;
  onRatingCountChange?: (count: number) => void;
  onRequireAuth?: () => void;
}

interface FilmDetailContentProps {
  filmId: number;
  initialFilm?: FilmListItem | null;
  isAuthenticated: boolean;
  authNextPath: string;
  titleId: string;
  readOnlyActions?: boolean;
  onFilmPatch?: (filmId: number, patch: Partial<FilmListItem>) => void;
  onRatingCountChange?: (count: number) => void;
  onRequireAuth?: () => void;
}

function toDetail(film: FilmListItem): FilmDetail {
  return { ...film, summary: null };
}

function FilmDetailContent({
  filmId,
  initialFilm,
  isAuthenticated,
  authNextPath,
  titleId,
  readOnlyActions = false,
  onFilmPatch,
  onRatingCountChange,
  onRequireAuth,
}: FilmDetailContentProps) {
  const [detail, setDetail] = useState<FilmDetail | null>(
    initialFilm ? toDetail(initialFilm) : null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetchFilmDetail(filmId, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) {
          setDetail(payload);
          setError(null);
        }
      })
      .catch((caught) => {
        if (!controller.signal.aborted) {
          setError(errorMessage(caught));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [attempt, filmId]);

  const patch = useCallback(
    (values: Partial<FilmListItem>) => {
      setDetail((current) =>
        current ? { ...current, ...values } : current,
      );
      onFilmPatch?.(filmId, values);
    },
    [filmId, onFilmPatch],
  );

  const handleRating = useCallback(
    async (rating: number) => {
      if (!isAuthenticated) {
        onRequireAuth?.();
        return;
      }
      setBusy(true);
      setActionError(null);
      try {
        const result = await setFilmRating(filmId, rating);
        patch({
          user_rating: result.rating,
          user_list_status: "seen",
        });
        onRatingCountChange?.(result.ratings_count);
      } catch (caught) {
        setActionError(errorMessage(caught));
      } finally {
        setBusy(false);
      }
    },
    [filmId, isAuthenticated, onRatingCountChange, onRequireAuth, patch],
  );

  const handleRatingDelete = useCallback(async () => {
    if (!isAuthenticated) {
      onRequireAuth?.();
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      const result = await deleteFilmRating(filmId);
      patch({ user_rating: null, user_list_status: null });
      onRatingCountChange?.(result.ratings_count);
    } catch (caught) {
      setActionError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }, [filmId, isAuthenticated, onRatingCountChange, onRequireAuth, patch]);

  const handleList = useCallback(
    async (status: EditableFilmListStatus) => {
      if (!isAuthenticated) {
        onRequireAuth?.();
        return;
      }
      setBusy(true);
      setActionError(null);
      try {
        await saveFilmListStatus(filmId, status);
        patch({ user_list_status: status });
      } catch (caught) {
        setActionError(errorMessage(caught));
      } finally {
        setBusy(false);
      }
    },
    [filmId, isAuthenticated, onRequireAuth, patch],
  );

  const handleListRemove = useCallback(async () => {
    if (!isAuthenticated) {
      onRequireAuth?.();
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      await removeFilmListStatus(filmId);
      patch({ user_list_status: null });
    } catch (caught) {
      setActionError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }, [filmId, isAuthenticated, onRequireAuth, patch]);

  const retry = () => {
    setLoading(true);
    setError(null);
    setAttempt((value) => value + 1);
  };

  if (detail) {
    return (
      <>
        <FilmDetailBody
          film={detail}
          titleId={titleId}
          isAuthenticated={isAuthenticated}
          authNextPath={authNextPath}
          onRatingChange={isAuthenticated ? handleRating : undefined}
          onRatingDelete={isAuthenticated ? handleRatingDelete : undefined}
          onListChange={isAuthenticated ? handleList : undefined}
          onListRemove={isAuthenticated ? handleListRemove : undefined}
          busy={busy}
          readOnlyActions={readOnlyActions}
          error={actionError ?? (loading ? null : error)}
          extra={
            <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-white/45">
              Film details
            </p>
          }
        />
        {loading ? (
          <span className="sr-only" role="status">
            Refreshing film details
          </span>
        ) : null}
      </>
    );
  }

  if (loading) {
    return (
      <div
        role="status"
        className="flex min-h-[24rem] flex-col items-center justify-center gap-3 text-white/55"
      >
        <LoaderCircle aria-hidden="true" className="h-5 w-5 animate-spin" />
        <p className="text-xs">Loading film details...</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-[24rem] flex-col items-center justify-center gap-4 px-6 text-center">
      <h2 id={titleId} className="text-2xl font-semibold text-white">
        Details unavailable
      </h2>
      <p className="text-sm text-white/55">
        {error ?? "This film could not be loaded."}
      </p>
      <button
        type="button"
        onClick={retry}
        className="inline-flex h-10 items-center gap-2 rounded-full bg-white px-5 text-xs font-medium text-black"
      >
        <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
        Try again
      </button>
    </div>
  );
}

export function FilmDetailDialog({
  open,
  filmId,
  initialFilm,
  isAuthenticated,
  authNextPath = "/films",
  readOnlyActions = false,
  onClose,
  onFilmPatch,
  onRatingCountChange,
  onRequireAuth,
}: FilmDetailDialogProps) {
  const titleId = useId();
  return (
    <Dialog
      open={open}
      onClose={onClose}
      titleId={titleId}
      className="max-w-2xl"
    >
      {open && filmId !== null ? (
        <FilmDetailContent
          key={filmId}
          filmId={filmId}
          initialFilm={initialFilm}
          isAuthenticated={isAuthenticated}
          authNextPath={authNextPath}
          titleId={titleId}
          readOnlyActions={readOnlyActions}
          onFilmPatch={onFilmPatch}
          onRatingCountChange={onRatingCountChange}
          onRequireAuth={onRequireAuth}
        />
      ) : null}
    </Dialog>
  );
}
