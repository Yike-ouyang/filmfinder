import Link from "next/link";
import { ArrowRight, UserRoundPlus } from "lucide-react";

export function BuildProfileCTA() {
  return (
    <div className="mx-auto mt-8 w-full max-w-xl rounded-[1.75rem] border border-white/10 bg-black/40 px-5 py-5 text-center shadow-[0_30px_80px_-35px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:px-7 sm:py-6">
      <div className="flex items-center justify-center gap-2">
        <UserRoundPlus
          aria-hidden="true"
          className="h-4 w-4 text-white/50"
        />
        <p className="text-sm font-medium text-white">
          No Letterboxd account?
        </p>
      </div>
      <p className="mx-auto mt-2 max-w-md text-balance text-xs leading-relaxed text-white/55 sm:text-[13px]">
        No problem! Create a FilmFinder account, rate the movies you&apos;ve
        watched, and discover recommendations tailored to your taste.
      </p>
      <Link
        href="/films"
        className="group mt-4 inline-flex h-10 items-center justify-center gap-2 rounded-full border border-white/15 bg-white/[0.03] px-5 text-sm font-medium text-white/80 transition-colors duration-300 hover:border-white/30 hover:bg-white/[0.08] hover:text-white"
      >
        Build My Movie Profile
        <ArrowRight
          aria-hidden="true"
          className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5"
        />
      </Link>
    </div>
  );
}
