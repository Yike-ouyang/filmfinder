import type { Metadata } from "next";
import { Suspense } from "react";
import { FilmsPageContent } from "@/components/films/FilmsPageContent";

export const metadata: Metadata = {
  title: "Films",
  description:
    "Browse the FilmFinder movie catalogue, rate films, and build your personal taste profile.",
};

function FilmsFallback() {
  return (
    <main
      id="main"
      className="relative flex min-h-svh items-center justify-center bg-background"
    >
      <div
        role="status"
        aria-label="Loading films"
        className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-white/70"
      />
    </main>
  );
}

export default function FilmsPage() {
  return (
    <Suspense fallback={<FilmsFallback />}>
      <FilmsPageContent />
    </Suspense>
  );
}
