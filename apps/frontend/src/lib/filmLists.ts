import { removeSwipe, saveSwipe } from "@/lib/swipes";
import type { FilmListStatus } from "@/types/films";

export type EditableFilmListStatus = Exclude<FilmListStatus, "seen">;

export async function saveFilmListStatus(
  filmId: number,
  status: EditableFilmListStatus,
): Promise<FilmListStatus> {
  await saveSwipe(
    filmId,
    status === "watchlist" ? "interested" : "not_interested",
  );
  return status;
}

export async function removeFilmListStatus(filmId: number): Promise<void> {
  await removeSwipe(filmId);
}
