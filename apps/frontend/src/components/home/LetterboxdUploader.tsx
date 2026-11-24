"use client";

import {
  AlertCircle,
  ArrowRight,
  FileArchive,
  Upload,
  X,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { UploadProgress } from "@/components/home/UploadProgress";
import {
  ApiError,
  errorMessage,
  requestRecommendations,
} from "@/lib/api";
import { cn } from "@/lib/cn";
import { LOADING_MESSAGES } from "@/lib/constants";
import { saveBatch } from "@/lib/storage";
import {
  formatBytes,
  uploadLimitLabel,
  validateZipFile,
} from "@/lib/upload";

type Phase = "idle" | "selected" | "uploading" | "ready";

interface LetterboxdUploaderProps {
  buttonLabel?: string;
}

export function LetterboxdUploader({
  buttonLabel = "Import Letterboxd",
}: LetterboxdUploaderProps = {}) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const redirectTimerRef = useRef<number | null>(null);
  const dragDepthRef = useRef(0);

  const [phase, setPhase] = useState<Phase>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    router.prefetch("/discover");
  }, [router]);

  useEffect(() => {
    if (phase !== "uploading") {
      return;
    }
    const interval = window.setInterval(() => {
      setMessageIndex((index) => index + 1);
    }, 2400);
    return () => window.clearInterval(interval);
  }, [phase]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (redirectTimerRef.current !== null) {
        window.clearTimeout(redirectTimerRef.current);
      }
    };
  }, []);

  const resetInput = () => {
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const chooseFile = useCallback((file: File | null) => {
    if (!file) {
      return;
    }
    const validation = validateZipFile(file);
    if (!validation.ok) {
      setSelectedFile(null);
      setPhase("idle");
      setError(validation.message);
      return;
    }
    setError(null);
    setSelectedFile(file);
    setPhase("selected");
    setMessageIndex(0);
  }, []);

  const handleInputChange = (file: File | undefined) => {
    chooseFile(file ?? null);
    resetInput();
  };

  const removeFile = () => {
    if (phase === "uploading" || phase === "ready") {
      return;
    }
    setSelectedFile(null);
    setError(null);
    setPhase("idle");
    resetInput();
  };

  const cancelUpload = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPhase("selected");
  };

  const startUpload = async () => {
    if (!selectedFile || phase === "uploading" || phase === "ready") {
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setMessageIndex(0);
    setPhase("uploading");

    try {
      const result = await requestRecommendations({
        file: selectedFile,
        signal: controller.signal,
      });

      if (controller.signal.aborted) {
        setPhase("selected");
        return;
      }

      if (!saveBatch(result, [])) {
        setError(
          "Your browser could not store this session. Allow session storage for this site and try again.",
        );
        setPhase("selected");
        return;
      }

      setPhase("ready");
      redirectTimerRef.current = window.setTimeout(() => {
        router.push("/discover");
      }, 700);
    } catch (uploadError) {
      if (uploadError instanceof ApiError && uploadError.code === "cancelled") {
        setPhase("selected");
        return;
      }
      setError(errorMessage(uploadError));
      setPhase("selected");
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  };

  const busy = phase === "uploading" || phase === "ready";
  const wide = phase !== "idle" || Boolean(error);
  const chooseFileRef = useRef(chooseFile);

  useEffect(() => {
    chooseFileRef.current = chooseFile;
  }, [chooseFile]);

  useEffect(() => {
    if (busy) {
      return;
    }

    const hasFiles = (event: Event) => {
      const dragEvent = event as globalThis.DragEvent;
      return Array.from(dragEvent.dataTransfer?.types ?? []).includes("Files");
    };

    const onDragEnter = (event: Event) => {
      if (!hasFiles(event)) {
        return;
      }
      event.preventDefault();
      dragDepthRef.current += 1;
      setDragging(true);
    };

    const onDragOver = (event: Event) => {
      if (!hasFiles(event)) {
        return;
      }
      event.preventDefault();
    };

    const onDragLeave = (event: Event) => {
      if (!hasFiles(event)) {
        return;
      }
      const dragEvent = event as globalThis.DragEvent;
      if (dragEvent.relatedTarget === null) {
        dragDepthRef.current = 0;
        setDragging(false);
        return;
      }
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (dragDepthRef.current === 0) {
        setDragging(false);
      }
    };

    const onDrop = (event: Event) => {
      if (!hasFiles(event)) {
        return;
      }
      event.preventDefault();
      const dragEvent = event as globalThis.DragEvent;
      dragDepthRef.current = 0;
      setDragging(false);
      chooseFileRef.current(dragEvent.dataTransfer?.files?.[0] ?? null);
    };

    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [busy]);

  return (
    <div
      className={cn(
        "flex flex-col items-center",
        wide ? "w-full max-w-md" : "w-full sm:w-auto",
      )}
      aria-busy={busy}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".zip,application/zip,application/x-zip-compressed"
        className="sr-only"
        tabIndex={-1}
        aria-label="Letterboxd ZIP export"
        onChange={(event) => handleInputChange(event.target.files?.[0])}
      />

      {phase === "idle" ? (
        <div className="flex w-full flex-col items-center">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="group inline-flex h-12 items-center gap-2.5 rounded-full bg-white px-7 text-sm font-medium text-black shadow-[0_20px_60px_-20px_rgba(255,255,255,0.35)] transition-[transform,opacity] duration-300 hover:opacity-90 active:scale-[0.98]"
          >
            <Upload
              aria-hidden="true"
              className="h-4 w-4 transition-transform duration-300 group-hover:-translate-y-0.5"
            />
            {buttonLabel}
            <ArrowRight
              aria-hidden="true"
              className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5"
            />
          </button>
          <p className="mt-3 text-[11px] text-white/50">
            Letterboxd ZIP export · max {uploadLimitLabel()}
          </p>
        </div>
      ) : null}

      {phase === "selected" && selectedFile ? (
        <div className="w-full rounded-[1.75rem] border border-white/10 bg-black/55 p-2 shadow-[0_30px_80px_-30px_rgba(0,0,0,0.9)] backdrop-blur-xl">
          <div className="flex items-center gap-3 px-3 py-2.5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full border border-white/10 bg-white/[0.04]">
              <FileArchive aria-hidden="true" className="h-4 w-4 text-white/70" />
            </span>
            <div className="min-w-0 flex-1 text-left">
              <p className="truncate text-sm font-medium text-white">
                {selectedFile.name}
              </p>
              <p className="mt-0.5 text-[11px] text-white/50">
                {formatBytes(selectedFile.size)} · ready to analyze
              </p>
            </div>
            <button
              type="button"
              onClick={removeFile}
              aria-label="Remove selected file"
              className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-white/55 transition-colors hover:bg-white/10 hover:text-white"
            >
              <X aria-hidden="true" className="h-4 w-4" />
            </button>
          </div>
          <button
            type="button"
            onClick={startUpload}
            className="group flex h-11 w-full items-center justify-center gap-2 rounded-[1.25rem] bg-white text-sm font-medium text-black transition-opacity hover:opacity-90"
          >
            Analyze my taste
            <ArrowRight
              aria-hidden="true"
              className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5"
            />
          </button>
        </div>
      ) : null}

      {error ? (
        <div
          role="alert"
          className="mt-4 flex w-full items-start gap-2.5 rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-3 text-left"
        >
          <AlertCircle
            aria-hidden="true"
            className="mt-0.5 h-4 w-4 shrink-0 text-danger"
          />
          <p className="text-xs leading-relaxed text-danger/90">{error}</p>
        </div>
      ) : null}

      {dragging ? (
        <div className="pointer-events-none fixed inset-0 z-[70] flex items-center justify-center bg-black/80 p-6 backdrop-blur-md">
          <div className="flex w-full max-w-lg flex-col items-center rounded-3xl border border-dashed border-white/25 px-8 py-16 text-center">
            <Upload aria-hidden="true" className="h-7 w-7 text-white/70" />
            <p className="mt-5 text-lg font-medium text-white">
              Drop your Letterboxd ZIP
            </p>
            <p className="mt-2 text-xs text-white/55">
              Max {uploadLimitLabel()} · files stay in memory
            </p>
          </div>
        </div>
      ) : null}

      <UploadProgress
        open={phase === "uploading" || phase === "ready"}
        phase={phase === "ready" ? "ready" : "uploading"}
        messageIndex={messageIndex % LOADING_MESSAGES.length}
        fileName={selectedFile?.name ?? "Letterboxd export"}
        onCancel={cancelUpload}
      />
    </div>
  );
}
