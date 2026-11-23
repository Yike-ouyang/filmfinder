"use client";

import { Search, SlidersHorizontal } from "lucide-react";
import type { FilmOrder, FilmSort } from "@/types/films";

export type SortChoice =
  | "popular"
  | "rating"
  | "newest"
  | "oldest";

interface FilmFiltersProps {
  search: string;
  onSearchChange: (value: string) => void;
  sort: FilmSort;
  order: FilmOrder;
  onSortChange: (sort: FilmSort, order: FilmOrder) => void;
  genre: string;
  onGenreChange: (genre: string) => void;
  genres: string[];
  temporalLimit: number;
  disabled?: boolean;
}

function currentChoice(sort: FilmSort, order: FilmOrder): SortChoice {
  if (sort === "rating") {
    return "rating";
  }
  if (sort === "release_date") {
    return order === "asc" ? "oldest" : "newest";
  }
  return "popular";
}

export function FilmFilters({
  search,
  onSearchChange,
  sort,
  order,
  onSortChange,
  genre,
  onGenreChange,
  genres,
  temporalLimit,
  disabled = false,
}: FilmFiltersProps) {
  const choice = currentChoice(sort, order);

  const handleSort = (value: SortChoice) => {
    if (value === "rating") {
      onSortChange("rating", "desc");
    } else if (value === "newest") {
      onSortChange("release_date", "desc");
    } else if (value === "oldest") {
      onSortChange("release_date", "asc");
    } else {
      onSortChange("popular", "desc");
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-white/10 bg-white/[0.025] px-4 py-3 text-xs leading-relaxed text-white/50">
        Our movie catalog currently includes films released up to{" "}
        {temporalLimit}.
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <label className="relative flex-1">
          <span className="sr-only">Search films by title</span>
          <Search
            aria-hidden="true"
            className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/35"
          />
          <input
            type="search"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            disabled={disabled}
            placeholder="Search films by title"
            className="h-11 w-full rounded-full border border-white/10 bg-black/35 pl-11 pr-4 text-sm text-white placeholder:text-white/35 focus:border-white/30 focus:outline-none disabled:opacity-50"
          />
        </label>

        <div className="flex flex-col gap-3 sm:flex-row">
          <label className="flex items-center gap-2">
            <SlidersHorizontal
              aria-hidden="true"
              className="h-4 w-4 shrink-0 text-white/35"
            />
            <span className="sr-only">Sort films</span>
            <select
              value={choice}
              onChange={(event) =>
                handleSort(event.target.value as SortChoice)
              }
              disabled={disabled}
              className="h-11 min-w-[10rem] rounded-full border border-white/10 bg-[#101010] px-4 text-xs text-white/75 focus:border-white/30 focus:outline-none disabled:opacity-50"
            >
              <option value="popular">Most Popular</option>
              <option value="rating">Highest Rated</option>
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
            </select>
          </label>

          <label>
            <span className="sr-only">Filter by genre</span>
            <select
              value={genre}
              onChange={(event) => onGenreChange(event.target.value)}
              disabled={disabled}
              className="h-11 min-w-[9rem] rounded-full border border-white/10 bg-[#101010] px-4 text-xs text-white/75 focus:border-white/30 focus:outline-none disabled:opacity-50"
            >
              <option value="">All genres</option>
              {genres.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
    </div>
  );
}
