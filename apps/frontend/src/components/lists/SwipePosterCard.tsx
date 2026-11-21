"use client";

import { Film, LoaderCircle, X } from "lucide-react";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import type { SwipeItem } from "@/types/swipes";

interface SwipePosterCardProps {
  item: SwipeItem;
  removing: boolean;
  onRemove: (item: SwipeItem) => void;
}

function PosterFallback({ title }: { title: string }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 bg-[radial-gradient(120%_100%_at_30%_0%,#242424,#101010_55%,#0a0a0a)] p-4 text-center">
      <Film aria-hidden="true" className="h-7 w-7 text-white/25" />
      <p className="text-balance text-xs font-medium text-white/65">
        {title}
      </p>
    </div>
  );
}

export function SwipePosterCard({
  item,
  removing,
  onRemove,
}: SwipePosterCardProps) {
  const genreLabel = item.genres.slice(0, 2).join(" · ");

  return (
    <article className="group">
      <div className="relative aspect-[2/3] w-full overflow-hidden rounded-xl border border-white/10 bg-surface shadow-[0_24px_60px_-30px_rgba(0,0,0,0.95)]">
        <TmdbMedia
          path={item.poster_path}
          variant="poster"
          alt={`${item.title}${item.year ? ` (${item.year})` : ""} poster`}
          sizes="(max-width: 640px) 45vw, 220px"
          fallback={<PosterFallback title={item.title} />}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,transparent_62%,rgba(0,0,0,0.55)_100%)]"
        />
        <button
          type="button"
          onClick={() => onRemove(item)}
          disabled={removing}
          aria-label={`Remove ${item.title} from this list`}
          title="Remove from this list"
          className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-full border border-white/15 bg-black/60 text-white/70 backdrop-blur-md transition-colors hover:bg-white/15 hover:text-white disabled:opacity-50"
        >
          {removing ? (
            <LoaderCircle
              aria-hidden="true"
              className="h-4 w-4 animate-spin"
            />
          ) : (
            <X aria-hidden="true" className="h-4 w-4" />
          )}
        </button>
      </div>
      <h3 className="mt-2.5 line-clamp-2 text-sm font-medium leading-snug text-white">
        {item.title}
      </h3>
      <p className="mt-1 truncate text-[11px] text-white/50">
        {[item.year, genreLabel].filter(Boolean).join(" · ") ||
          "Film"}
      </p>
    </article>
  );
}
