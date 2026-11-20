"use client";

import { ArrowUpRight, Info } from "lucide-react";
import { useId, useState } from "react";
import { Dialog } from "@/components/ui/Dialog";

const steps = [
  {
    number: "01",
    title: "Export your Letterboxd data",
    body: "Open Letterboxd settings, go to Data, and create a fresh export. Keep the downloaded ZIP as-is.",
  },
  {
    number: "02",
    title: "Let FilmFinder read your taste",
    body: "Your ratings and watched films are matched to the catalogue, then the M5 model predicts what you might love next.",
  },
  {
    number: "03",
    title: "Swipe through the picks",
    body: "Keep what interests you, pass on the rest, or mark films you have already seen. Your decisions stay in this browser.",
  },
];

export function HowItWorks() {
  const [open, setOpen] = useState(false);
  const titleId = useId();

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group inline-flex h-12 items-center gap-2 rounded-full border border-white/15 bg-white/[0.03] px-6 text-sm font-medium text-white/75 backdrop-blur-md transition-colors duration-300 hover:border-white/25 hover:bg-white/[0.07] hover:text-white"
      >
        <Info aria-hidden="true" className="h-4 w-4 text-white/50" />
        How it works
      </button>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        titleId={titleId}
        className="max-w-xl"
      >
        <div className="p-7 pb-8 pt-16 sm:p-10 sm:pt-16">
          <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
            How it works
          </p>
          <h2
            id={titleId}
            className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl"
          >
            From ratings to your next watch.
          </h2>

          <ol className="mt-9 space-y-7">
            {steps.map((step) => (
              <li key={step.number} className="flex gap-4">
                <span className="mt-0.5 font-mono text-[11px] text-white/50">
                  {step.number}
                </span>
                <div>
                  <h3 className="text-sm font-medium text-white">
                    {step.title}
                  </h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-white/50">
                    {step.body}
                  </p>
                </div>
              </li>
            ))}
          </ol>

          <div className="mt-9 flex flex-col gap-4 border-t border-white/10 pt-6 sm:flex-row sm:items-center sm:justify-between">
            <a
              href="https://letterboxd.com/settings/data/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-white/50 transition-colors hover:text-white"
            >
              Open Letterboxd export settings
              <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
            </a>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="inline-flex h-10 items-center justify-center rounded-full bg-white px-5 text-sm font-medium text-black transition-opacity hover:opacity-85"
            >
              Got it
            </button>
          </div>
        </div>
      </Dialog>
    </>
  );
}
