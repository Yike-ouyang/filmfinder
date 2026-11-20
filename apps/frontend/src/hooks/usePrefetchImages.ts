"use client";

import { useEffect } from "react";
import { getTmdbBackdropUrl, getTmdbPosterUrl } from "@/lib/tmdb";
import type { Recommendation } from "@/types/recommendations";

const PREFETCH_COUNT = 3;

export function usePrefetchImages(
  recommendations: Recommendation[],
  currentIndex: number,
): void {
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    for (let offset = 1; offset <= PREFETCH_COUNT; offset += 1) {
      const next = recommendations[currentIndex + offset];
      if (!next) {
        break;
      }
      for (const url of [
        getTmdbPosterUrl(next.poster_path),
        getTmdbBackdropUrl(next.backdrop_path),
      ]) {
        if (!url) {
          continue;
        }
        const image = new window.Image();
        image.decoding = "async";
        image.src = url;
      }
    }
  }, [recommendations, currentIndex]);
}
