"use client";

import { RotateCcw, SearchX } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { FilmDetailDialog } from "@/components/films/FilmDetailDialog";
import { FilmFilters } from "@/components/films/FilmFilters";
import { FilmGrid } from "@/components/films/FilmGrid";
import { FilmPagination } from "@/components/films/FilmPagination";
import { ProfileProgress } from "@/components/films/ProfileProgress";
import { TmdbAttribution } from "@/components/ui/TmdbAttribution";
import { ApiError, errorMessage } from "@/lib/api";
import {
  fetchFilmFacets,
  fetchFilms,
} from "@/lib/films";
import { fetchProfileSummary } from "@/lib/ratings";
import { requestManualRecommendationGeneration } from "@/lib/serverRecommendations";
import { FILM_CATALOG_YEAR_LIMIT } from "@/lib/constants";
import type {
  FilmFacets,
  FilmListItem,
  FilmListResponse,
  FilmOrder,
  FilmSort,
  ProfileSummary,
} from "@/types/films";

type LoadStatus = "loading" | "ready" | "error";

function parsePage(value: string | null): number {
  const parsed = Number(value ?? "1");
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 1;
}

function parseSort(value: string | null): FilmSort {
  if (value === "rating" || value === "release_date") {
    return value;
  }
  return "popular";
}

function SkeletonGrid() {
  return (
    <div
      role="status"
      aria-label="Loading films"
      className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
    >
      {Array.from({ length: 12 }, (_, index) => (
        <div key={index} className="animate-pulse">
          <div className="aspect-[2/3] w-full rounded-xl bg-white/[0.05]" />
          <div className="mt-3 h-3 w-4/5 rounded bg-white/[0.06]" />
          <div className="mt-2 h-2.5 w-1/3 rounded bg-white/[0.04]" />
        </div>
      ))}
    </div>
  );
}

export function FilmsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, loading: authLoading } = useAuth();

  const queryPage = parsePage(searchParams.get("page"));
  const querySearch = (searchParams.get("q") ?? "").trim();
  const querySort = parseSort(searchParams.get("sort"));
  const queryOrder: FilmOrder =
    querySort === "release_date" &&
    searchParams.get("order") === "asc"
      ? "asc"
      : "desc";
  const queryGenre = searchParams.get("genre") ?? "";
  const rawFilm = searchParams.get("film");
  const parsedFilm = rawFilm ? Number(rawFilm) : NaN;
  const selectedFilmId =
    Number.isInteger(parsedFilm) && parsedFilm >= 0 ? parsedFilm : null;

  const [searchInput, setSearchInput] = useState(querySearch);
  const [list, setList] = useState<FilmListResponse | null>(null);
  const [listStatus, setListStatus] = useState<LoadStatus>("loading");
  const [listError, setListError] = useState<string | null>(null);
  const [listAttempt, setListAttempt] = useState(0);
  const [facets, setFacets] = useState<FilmFacets | null>(null);
  const [summary, setSummary] = useState<ProfileSummary | null>(null);
  const [summaryUserId, setSummaryUserId] = useState<number | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const updateQuery = useCallback(
    (
      updates: Record<string, string | null>,
      options: { replace?: boolean } = {},
    ) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(updates)) {
        if (value === null || value === "") {
          next.delete(key);
        } else {
          next.set(key, value);
        }
      }
      const query = next.toString();
      const url = query ? `/films?${query}` : "/films";
      if (options.replace) {
        router.replace(url, { scroll: false });
      } else {
        router.push(url, { scroll: false });
      }
    },
    [router, searchParams],
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearchInput(querySearch);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [querySearch]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const value = searchInput.trim();
      if (value === querySearch) {
        return;
      }
      updateQuery(
        {
          q: value || null,
          page: null,
          film: null,
        },
        { replace: true },
      );
    }, 300);
    return () => window.clearTimeout(timer);
  }, [querySearch, searchInput, updateQuery]);

  useEffect(() => {
    const controller = new AbortController();
    fetchFilmFacets(controller.signal)
      .then((payload) => setFacets(payload))
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const timer = window.setTimeout(() => {
      if (!active) {
        return;
      }
      setListStatus("loading");
      setListError(null);
      fetchFilms(
        {
          page: queryPage,
          search: querySearch,
          sort: querySort,
          order: queryOrder,
          genre: queryGenre,
        },
        controller.signal,
      )
        .then((payload) => {
          if (!controller.signal.aborted) {
            setList(payload);
            setListStatus("ready");
          }
        })
        .catch((error) => {
          if (controller.signal.aborted) {
            return;
          }
          if (error instanceof ApiError && error.code === "cancelled") {
            return;
          }
          setListError(errorMessage(error));
          setListStatus("error");
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    listAttempt,
    queryGenre,
    queryOrder,
    queryPage,
    querySearch,
    querySort,
  ]);

  useEffect(() => {
    if (!user) {
      return;
    }
    const controller = new AbortController();
    let active = true;
    const timer = window.setTimeout(() => {
      if (!active) {
        return;
      }
      setSummaryLoading(true);
      fetchProfileSummary()
        .then((payload) => {
          if (!controller.signal.aborted) {
            setSummary(payload);
            setSummaryUserId(user.id);
          }
        })
        .catch(() => undefined)
        .finally(() => {
          if (!controller.signal.aborted) {
            setSummaryLoading(false);
          }
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [user]);

  const handleSortChange = useCallback(
    (sort: FilmSort, order: FilmOrder) => {
      updateQuery({
        sort: sort === "popular" ? null : sort,
        order:
          sort === "release_date" && order === "asc"
            ? "asc"
            : null,
        page: null,
        film: null,
      });
    },
    [updateQuery],
  );

  const handleGenreChange = useCallback(
    (genre: string) => {
      updateQuery({
        genre: genre || null,
        page: null,
        film: null,
      });
    },
    [updateQuery],
  );

  const handlePage = useCallback(
    (page: number) => {
      updateQuery({
        page: page > 1 ? String(page) : null,
        film: null,
      });
    },
    [updateQuery],
  );

  const handleOpenFilm = useCallback(
    (filmId: number) => {
      updateQuery({ film: String(filmId) });
    },
    [updateQuery],
  );

  const handleCloseFilm = useCallback(() => {
    updateQuery({ film: null });
  }, [updateQuery]);

  const handleFilmPatch = useCallback(
    (filmId: number, patch: Partial<FilmListItem>) => {
      setList((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          items: current.items.map((film) =>
            film.internal_movie_id === filmId
              ? { ...film, ...patch }
              : film,
          ),
        };
      });
    },
    [],
  );

  const handleRatingCountChange = useCallback((count: number) => {
    setSummary((current) =>
      current
        ? {
            ...current,
            ratings_count: count,
            can_generate:
              count >= current.recommended_min_ratings,
          }
        : current,
    );
    if (user) {
      setSummaryUserId(user.id);
    }
  }, [user]);

  const handleGenerate = useCallback(async () => {
    if (!user || generating) {
      return;
    }
    setGenerating(true);
    setGenerateError(null);
    try {
      await requestManualRecommendationGeneration();
      router.push("/discover");
    } catch (error) {
      setGenerateError(errorMessage(error));
    } finally {
      setGenerating(false);
    }
  }, [generating, router, user]);

  const handleClearFilters = useCallback(() => {
    setSearchInput("");
    updateQuery({
      q: null,
      genre: null,
      page: null,
      film: null,
    });
  }, [updateQuery]);

  const initialFilm =
    selectedFilmId !== null
      ? list?.items.find(
          (film) => film.internal_movie_id === selectedFilmId,
        ) ?? null
      : null;

  const total = list?.total ?? 0;
  const showEmpty = listStatus === "ready" && total === 0;
  const visibleSummary =
    user && summaryUserId === user.id ? summary : null;
  const hasFilters = Boolean(
    querySearch || queryGenre || querySort !== "popular",
  );

  return (
    <main
      id="main"
      className="relative min-h-svh overflow-hidden bg-background px-5 pb-20 pt-28 sm:px-8"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_70%_at_50%_0%,#1a1a1a_0%,#0d0d0d_46%,#080808_82%)]"
      />
      <div className="relative z-10 mx-auto w-full max-w-7xl">
        <p className="text-[10px] font-medium uppercase tracking-[0.32em] text-white/45">
          Film catalogue
        </p>
        <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em] text-white sm:text-5xl">
          Films
        </h1>
        <p className="mt-3 max-w-xl text-sm leading-relaxed text-white/50">
          Browse the FilmFinder catalogue, rate what you have seen, and build
          a profile that powers your recommendations.
        </p>

        <div className="mt-7">
          <ProfileProgress
            summary={visibleSummary}
            loading={summaryLoading || authLoading}
            generating={generating}
            error={generateError}
            isAuthenticated={Boolean(user)}
            onGenerate={() => void handleGenerate()}
          />
        </div>

        <div className="mt-8">
          <FilmFilters
            search={searchInput}
            onSearchChange={setSearchInput}
            sort={querySort}
            order={queryOrder}
            onSortChange={handleSortChange}
            genre={queryGenre}
            onGenreChange={handleGenreChange}
            genres={facets?.genres ?? []}
            temporalLimit={FILM_CATALOG_YEAR_LIMIT}
          />
        </div>

        <div className="mt-8">
          {listStatus === "error" ? (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] px-6 py-12 text-center">
              <p className="text-sm font-medium text-white/70">
                The film catalogue could not be loaded.
              </p>
              <p className="mt-2 text-xs text-white/40">
                {listError ?? "Try again in a moment."}
              </p>
              <button
                type="button"
                onClick={() =>
                  setListAttempt((attempt) => attempt + 1)
                }
                className="mt-5 inline-flex h-10 items-center gap-2 rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-90"
              >
                <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
                Try again
              </button>
            </div>
          ) : listStatus === "loading" && !list ? (
            <SkeletonGrid />
          ) : showEmpty ? (
            <div className="rounded-3xl border border-white/10 bg-white/[0.02] px-6 py-16 text-center">
              <SearchX
                aria-hidden="true"
                className="mx-auto h-6 w-6 text-white/30"
              />
              <p className="mt-4 text-sm font-medium text-white/70">
                No films match your search.
              </p>
              <p className="mt-2 text-xs text-white/40">
                Try a different title or clear the filters.
              </p>
              {hasFilters ? (
                <button
                  type="button"
                  onClick={handleClearFilters}
                  className="mt-6 inline-flex h-10 items-center justify-center rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-90"
                >
                  Clear filters
                </button>
              ) : null}
            </div>
          ) : list ? (
            <div className={listStatus === "loading" ? "opacity-60" : undefined}>
              <FilmGrid films={list.items} onOpen={handleOpenFilm} />
              <div className="mt-10">
                <FilmPagination
                  page={list.page}
                  totalPages={list.total_pages}
                  total={list.total}
                  onPage={handlePage}
                  disabled={listStatus === "loading"}
                />
              </div>
            </div>
          ) : null}
        </div>

        <footer className="mt-16 text-center">
          <TmdbAttribution />
        </footer>
      </div>

      <FilmDetailDialog
        open={selectedFilmId !== null}
        filmId={selectedFilmId}
        initialFilm={initialFilm}
        isAuthenticated={Boolean(user)}
        authNextPath="/films"
        onClose={handleCloseFilm}
        onFilmPatch={handleFilmPatch}
        onRatingCountChange={handleRatingCountChange}
      />
    </main>
  );
}
