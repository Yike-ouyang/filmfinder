"use client";

import { FilmCard } from "@/components/films/FilmCard";
import type { FilmListItem } from "@/types/films";

interface FilmGridProps {
  films: FilmListItem[];
  onOpen: (filmId: number) => void;
}

export function FilmGrid({ films, onOpen }: FilmGridProps) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
      {films.map((film) => (
        <FilmCard
          key={film.internal_movie_id}
          film={film}
          onOpen={onOpen}
        />
      ))}
    </div>
  );
}
