import { Clock, Film, HardDrive, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { TmdbAttribution } from "@/components/ui/TmdbAttribution";
import type { BatchLoadStatus } from "@/types/recommendations";

const copy: Record<
  Exclude<BatchLoadStatus, "ready">,
  { title: string; body: string; icon: typeof Film }
> = {
  missing: {
    title: "No discoveries yet.",
    body: "Import your Letterboxd export to start a personalized session.",
    icon: Film,
  },
  expired: {
    title: "Your discovery session expired.",
    body: "Recommendations are kept in this browser for 24 hours. Import your profile again to start a new session.",
    icon: Clock,
  },
  invalid: {
    title: "This session could not be restored.",
    body: "The saved recommendation batch was incomplete or damaged. Start a fresh import.",
    icon: TriangleAlert,
  },
  unavailable: {
    title: "Session storage is unavailable.",
    body: "This browser is blocking local session storage. Allow it for FilmFinder, then import your profile.",
    icon: HardDrive,
  },
};

export function EmptyDiscovery({
  status,
}: {
  status: Exclude<BatchLoadStatus, "ready">;
}) {
  const content = copy[status];
  const Icon = content.icon;

  return (
    <main
      id="main"
      className="relative flex min-h-svh flex-col overflow-hidden bg-background"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_80%_at_50%_0%,#1b1b1b_0%,#0d0d0d_45%,#080808_80%)]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(70%_50%_at_50%_45%,rgba(255,255,255,0.03),transparent_70%)]"
      />
      <div className="relative z-10 flex flex-1 items-center justify-center px-6 pb-20 pt-28">
        <div className="w-full max-w-md text-center">
          <span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-white/10 bg-white/[0.04]">
            <Icon aria-hidden="true" className="h-5 w-5 text-white/60" />
          </span>
          <h1 className="mt-6 text-3xl font-semibold tracking-[-0.035em] text-white sm:text-4xl">
            {content.title}
          </h1>
          <p className="mt-4 text-balance text-sm leading-relaxed text-white/55">
            {content.body}
          </p>
          <Link
            href="/"
            className="mt-8 inline-flex h-11 items-center justify-center rounded-full bg-white px-6 text-sm font-medium text-black transition-opacity hover:opacity-90"
          >
            Import Letterboxd
          </Link>
        </div>
      </div>
      <footer className="safe-bottom relative z-10 px-6 text-center">
        <TmdbAttribution />
      </footer>
    </main>
  );
}
