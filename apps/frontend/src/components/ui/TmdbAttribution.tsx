import { cn } from "@/lib/cn";

export function TmdbAttribution({ className }: { className?: string }) {
  return (
    <p className={cn("text-[10px] tracking-wide text-white/50", className)}>
      Movie imagery and metadata{" "}
      <a
        href="https://www.themoviedb.org/"
        target="_blank"
        rel="noopener noreferrer"
        className="underline decoration-white/20 underline-offset-2 transition-colors hover:text-white/60"
      >
        courtesy of TMDB
      </a>
      . This product is not endorsed by TMDB.
    </p>
  );
}
