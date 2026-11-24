"use client";

import Link from "next/link";
import {
  ArrowRight,
  ChevronRight,
  Compass,
  FileArchive,
  Library,
  Sparkles,
} from "lucide-react";
import { useId, useState } from "react";
import { HowItWorks } from "@/components/home/HowItWorks";
import { LetterboxdUploader } from "@/components/home/LetterboxdUploader";
import { Dialog } from "@/components/ui/Dialog";

const cardClass =
  "flex h-full flex-col rounded-3xl border border-white/10 bg-black/40 p-5 text-center shadow-[0_30px_80px_-35px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:p-6";

const iconClass =
  "mx-auto grid h-11 w-11 place-items-center rounded-full border border-white/10 bg-white/[0.04]";

export function StartOptions() {
  const [open, setOpen] = useState(false);
  const titleId = useId();

  return (
    <section
      aria-labelledby="start-options-title"
      className="mx-auto mt-10 w-full max-w-3xl text-left"
    >
      <h2 id="start-options-title" className="sr-only">
        How would you like to start?
      </h2>

      <article className={cardClass}>
        <span className={iconClass}>
          <FileArchive aria-hidden="true" className="h-5 w-5 text-white/70" />
        </span>
        <h3 className="mt-4 text-lg font-semibold tracking-[-0.02em] text-white">
          Import Letterboxd
        </h3>
        <p className="mt-2 text-xs leading-relaxed text-white/55">
          Already using Letterboxd? Import your movie history and get
          personalized recommendations.
        </p>
        <div className="mt-5 flex flex-1 flex-col items-center justify-center">
          <LetterboxdUploader buttonLabel="Import My Movies" />
        </div>
        <div className="mt-4 flex justify-center">
          <HowItWorks />
        </div>
      </article>

      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        className="group mt-4 flex w-full items-center gap-4 rounded-3xl border border-white/10 bg-black/40 p-5 text-left shadow-[0_30px_80px_-35px_rgba(0,0,0,0.95)] backdrop-blur-xl transition-colors duration-300 hover:border-white/25 hover:bg-black/55 sm:p-6"
      >
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full border border-white/10 bg-white/[0.04]">
          <Compass aria-hidden="true" className="h-5 w-5 text-white/70" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-white">
            No account?
          </span>
          <span className="mt-1 block text-xs leading-relaxed text-white/50">
            Choose how you&apos;d like to indicate your preferences.
          </span>
        </span>
        <ChevronRight
          aria-hidden="true"
          className="h-4 w-4 shrink-0 text-white/40 transition-transform duration-300 group-hover:translate-x-0.5 group-hover:text-white/70"
        />
      </button>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        titleId={titleId}
        className="max-w-2xl"
      >
        <div className="p-7 pt-16 sm:p-8 sm:pt-16">
          <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
            Build your profile
          </p>
          <h2
            id={titleId}
            className="mt-4 text-balance text-2xl font-semibold tracking-[-0.03em] text-white sm:text-3xl"
          >
            Choose how you&apos;d like to indicate your preferences
          </h2>

          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <article className={cardClass}>
              <span className={iconClass}>
                <Library
                  aria-hidden="true"
                  className="h-5 w-5 text-white/70"
                />
              </span>
              <h3 className="mt-4 text-lg font-semibold tracking-[-0.02em] text-white">
                Browse Movie Library
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-white/55">
                Explore our movie library, rate films you&apos;ve watched, and
                build your personal movie profile.
              </p>
              <div className="mt-5 flex flex-1 items-end justify-center">
                <Link
                  href="/films"
                  className="group inline-flex h-10 items-center justify-center gap-2 rounded-full bg-white px-5 text-sm font-medium text-black transition-opacity hover:opacity-90"
                >
                  Browse Movies
                  <ArrowRight
                    aria-hidden="true"
                    className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5"
                  />
                </Link>
              </div>
            </article>

            <article className={cardClass}>
              <span className={iconClass}>
                <Sparkles
                  aria-hidden="true"
                  className="h-5 w-5 text-white/70"
                />
              </span>
              <h3 className="mt-4 text-lg font-semibold tracking-[-0.02em] text-white">
                Quick Taste Profile
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-white/55">
                Not sure where to start? Rate 20 selected movies and discover
                your personal taste.
              </p>
              <div className="mt-5 flex flex-1 items-end justify-center">
                <Link
                  href="/taste-profile"
                  className="group inline-flex h-10 items-center justify-center gap-2 rounded-full bg-white px-5 text-sm font-medium text-black transition-opacity hover:opacity-90"
                >
                  Rate 20 Movies
                  <ArrowRight
                    aria-hidden="true"
                    className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5"
                  />
                </Link>
              </div>
            </article>
          </div>
        </div>
      </Dialog>
    </section>
  );
}
