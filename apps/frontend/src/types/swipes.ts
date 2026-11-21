export type SwipeAction = "interested" | "not_interested" | "already_seen";

export interface SwipeItem {
  internal_movie_id: number;
  movie_id: string;
  title: string;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  poster_path: string | null;
  backdrop_path: string | null;
  action: SwipeAction;
  updated_at: string;
}

export interface SwipeLists {
  interested: SwipeItem[];
  not_interested: SwipeItem[];
  counts: {
    interested: number;
    not_interested: number;
  };
}

export interface SwipeMutation {
  swipe: SwipeItem;
  event_id: number;
}

export interface SwipeUndoResult {
  swipe: SwipeItem | null;
}
