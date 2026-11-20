import { HowItWorks } from "@/components/home/HowItWorks";
import { LetterboxdUploader } from "@/components/home/LetterboxdUploader";
import { TmdbAttribution } from "@/components/ui/TmdbAttribution";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import {
  HERO_BACKDROP_PATH,
  SITE_DESCRIPTION,
  SITE_TAGLINE,
} from "@/lib/constants";

function HeroFallback() {
  return (
    <div
      aria-hidden="true"
      className="h-full w-full bg-[radial-gradient(120%_90%_at_70%_10%,#262626_0%,#111111_38%,#080808_78%)]"
    />
  );
}

export function HeroSection() {
  return (
    <section className="relative isolate flex min-h-svh flex-col overflow-hidden">
      <div aria-hidden="true" className="absolute inset-0">
        <TmdbMedia
          path={HERO_BACKDROP_PATH}
          variant="backdrop"
          alt=""
          sizes="100vw"
          preload
          imageClassName="scale-[1.03] object-[center_38%] brightness-[0.72] saturate-[0.85]"
          fallback={<HeroFallback />}
        />
      </div>
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[linear-gradient(180deg,rgba(8,8,8,0.62)_0%,rgba(8,8,8,0.22)_30%,rgba(8,8,8,0.48)_62%,#080808_100%)]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(85%_65%_at_50%_40%,transparent_0%,rgba(8,8,8,0.4)_70%,rgba(8,8,8,0.85)_100%)]"
      />

      <div className="relative z-10 flex flex-1 items-center justify-center px-5 pb-28 pt-32 sm:px-8 sm:pt-36">
        <div className="w-full max-w-4xl text-center">
          <p className="mb-7 text-[10px] font-medium uppercase tracking-[0.34em] text-white/50 sm:text-[11px]">
            Personalized movie discovery
          </p>
          <h1 className="text-balance text-[clamp(2.9rem,9vw,7rem)] font-semibold leading-[0.94] tracking-[-0.045em] text-white">
            {SITE_TAGLINE}
          </h1>
          <p className="mx-auto mt-7 max-w-xl text-balance text-sm leading-relaxed text-white/60 sm:text-base">
            {SITE_DESCRIPTION}
          </p>

          <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row sm:items-start sm:gap-5">
            <LetterboxdUploader />
            <HowItWorks />
          </div>
        </div>
      </div>

      <footer className="safe-bottom relative z-10 mx-auto flex w-full max-w-5xl flex-col items-center gap-3 px-5 text-center sm:px-8">
        <p className="text-[11px] tracking-wide text-white/50">
          No account required. Your archive is processed in memory.
        </p>
        <TmdbAttribution />
      </footer>
    </section>
  );
}
