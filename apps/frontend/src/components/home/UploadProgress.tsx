"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Check, LoaderCircle } from "lucide-react";
import { useId } from "react";
import { Dialog } from "@/components/ui/Dialog";
import { LOADING_MESSAGES } from "@/lib/constants";

interface UploadProgressProps {
  open: boolean;
  phase: "uploading" | "ready";
  messageIndex: number;
  fileName: string;
  onCancel: () => void;
}

export function UploadProgress({
  open,
  phase,
  messageIndex,
  fileName,
  onCancel,
}: UploadProgressProps) {
  const titleId = useId();
  const reduceMotion = useReducedMotion();
  const uploading = phase === "uploading";

  return (
    <Dialog
      open={open}
      onClose={uploading ? onCancel : () => undefined}
      titleId={titleId}
      showClose={uploading}
      closeLabel="Cancel upload"
      className="max-w-md"
    >
      <div
        className="flex flex-col items-center px-8 pb-9 pt-16 text-center sm:px-10"
        aria-busy={uploading}
      >
        <div className="relative grid h-16 w-16 place-items-center rounded-full border border-white/10 bg-white/[0.03]">
          {uploading ? (
            <LoaderCircle
              aria-hidden="true"
              className="h-6 w-6 animate-spin text-white/80"
            />
          ) : (
            <Check aria-hidden="true" className="h-6 w-6 text-white" />
          )}
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full ring-1 ring-white/5"
          />
        </div>

        <p className="mt-7 text-[10px] font-medium uppercase tracking-[0.3em] text-white/50">
          {uploading ? "Analyzing profile" : "Ready"}
        </p>
        <h2
          id={titleId}
          className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white"
        >
          {uploading ? "Finding your next films" : "Your discoveries are ready"}
        </h2>

        {uploading ? (
          <>
            <div
              aria-hidden="true"
              className="mt-6 h-7 w-full overflow-hidden text-sm text-white/55"
            >
              <AnimatePresence mode="wait" initial={false}>
                <motion.p
                  key={messageIndex}
                  initial={reduceMotion ? false : { opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -8 }}
                  transition={{ duration: reduceMotion ? 0 : 0.3 }}
                >
                  {LOADING_MESSAGES[messageIndex % LOADING_MESSAGES.length]}
                </motion.p>
              </AnimatePresence>
            </div>
            <p className="sr-only" role="status">
              Analyzing your Letterboxd profile.
            </p>
            <div
              aria-hidden="true"
              className="relative mt-5 h-px w-full overflow-hidden rounded-full bg-white/10"
            >
              <motion.span
                className="absolute inset-y-0 w-1/3 bg-white/70"
                animate={
                  reduceMotion
                    ? undefined
                    : { x: ["-110%", "320%"], opacity: [0.2, 1, 0.2] }
                }
                transition={
                  reduceMotion
                    ? undefined
                    : { duration: 1.8, repeat: Infinity, ease: "easeInOut" }
                }
                style={reduceMotion ? { left: "33%" } : undefined}
              />
            </div>
            <p className="mt-6 max-w-xs truncate text-xs text-white/45">
              {fileName}
            </p>
            <button
              type="button"
              onClick={onCancel}
              className="mt-7 text-xs text-white/55 underline decoration-white/20 underline-offset-4 transition-colors hover:text-white/70"
            >
              Cancel upload
            </button>
          </>
        ) : (
          <p className="mt-4 text-sm text-white/50">
            Taking you to your discoveries...
          </p>
        )}
      </div>
    </Dialog>
  );
}
