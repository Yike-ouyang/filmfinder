"use client";

import { ArrowUpRight, Info } from "lucide-react";
import { useId, useState } from "react";
import { Dialog } from "@/components/ui/Dialog";

const EXPORT_URL = "https://letterboxd.com/user/exportdata/";

const steps = [
  {
    number: "01",
    title: "Log in to your Letterboxd account",
    body: "FilmFinder only needs your public ratings and watched films. Sign in on Letterboxd first.",
  },
  {
    number: "02",
    title: "Open the Letterboxd data export page",
    body: "Go to letterboxd.com/user/exportdata/ and request a fresh export of your account data.",
  },
  {
    number: "03",
    title: "Download your data as a ZIP file",
    body: "Letterboxd sends or downloads a ZIP archive. Keep the archive exactly as it was created.",
  },
  {
    number: "04",
    title: "Return to FilmFinder",
    body: "Come back to this page and choose Import Letterboxd.",
  },
  {
    number: "05",
    title: "Upload the entire ZIP file",
    body: "Drop the complete ZIP archive. Do not extract it and do not upload the individual CSV files.",
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
            Letterboxd import
          </p>
          <h2
            id={titleId}
            className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl"
          >
            How to import your Letterboxd history
          </h2>

          <ol className="mt-8 space-y-6">
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

          <div className="mt-7 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3.5">
            <p className="text-xs font-medium text-white/75">
              Upload the entire ZIP file.
            </p>
            <p className="mt-1 text-xs leading-relaxed text-white/45">
              Do not unzip it or upload individual CSV files.
            </p>
          </div>

          <div className="mt-8 flex flex-col gap-4 border-t border-white/10 pt-6 sm:flex-row sm:items-center sm:justify-between">
            <a
              href={EXPORT_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-10 items-center justify-center gap-1.5 rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-85"
            >
              Export My Letterboxd Data
              <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
            </a>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="inline-flex h-10 items-center justify-center rounded-full border border-white/10 px-5 text-xs font-medium text-white/65 transition-colors hover:bg-white/5 hover:text-white"
            >
              Got it
            </button>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-white/35">
            If you are not signed in to Letterboxd, the export page will ask
            you to log in first.
          </p>
        </div>
      </Dialog>
    </>
  );
}
