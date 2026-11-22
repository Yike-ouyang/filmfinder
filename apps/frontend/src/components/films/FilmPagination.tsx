"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface FilmPaginationProps {
  page: number;
  totalPages: number;
  total: number;
  onPage: (page: number) => void;
  disabled?: boolean;
}

export function FilmPagination({
  page,
  totalPages,
  total,
  onPage,
  disabled = false,
}: FilmPaginationProps) {
  return (
    <nav
      aria-label="Film catalogue pagination"
      className="flex flex-col items-center justify-between gap-4 border-t border-white/10 pt-6 sm:flex-row"
    >
      <p className="text-xs text-white/45">
        {total.toLocaleString("en-US")} films
        <span className="mx-2 text-white/20">·</span>
        Page <span className="text-white/75">{page}</span> of{" "}
        {totalPages}
      </p>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onPage(page - 1)}
          disabled={disabled || page <= 1}
          className="inline-flex h-10 items-center gap-1.5 rounded-full border border-white/10 px-4 text-xs font-medium text-white/65 transition-colors hover:bg-white/5 hover:text-white disabled:pointer-events-none disabled:opacity-30"
        >
          <ChevronLeft aria-hidden="true" className="h-4 w-4" />
          Previous
        </button>
        <button
          type="button"
          onClick={() => onPage(page + 1)}
          disabled={disabled || page >= totalPages}
          className="inline-flex h-10 items-center gap-1.5 rounded-full border border-white/10 px-4 text-xs font-medium text-white/65 transition-colors hover:bg-white/5 hover:text-white disabled:pointer-events-none disabled:opacity-30"
        >
          Next
          <ChevronRight aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>
    </nav>
  );
}
