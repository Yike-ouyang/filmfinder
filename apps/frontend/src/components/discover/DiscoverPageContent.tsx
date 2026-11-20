"use client";

import { LoaderCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { DiscoverExperience } from "@/components/discover/DiscoverExperience";
import { EmptyDiscovery } from "@/components/discover/EmptyDiscovery";
import { loadBatch, type BatchLoadResult } from "@/lib/storage";

function DiscoverLoading() {
  return (
    <main
      id="main"
      className="relative flex min-h-svh items-center justify-center bg-background px-6"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(90%_70%_at_50%_0%,#1a1a1a_0%,#0c0c0c_45%,#080808_80%)]"
      />
      <div
        role="status"
        aria-label="Loading recommendations"
        className="relative z-10 flex flex-col items-center gap-4 text-white/55"
      >
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin text-white/60"
        />
        <p className="text-[10px] uppercase tracking-[0.28em]">
          Preparing discoveries
        </p>
      </div>
    </main>
  );
}

export function DiscoverPageContent() {
  const [result, setResult] = useState<BatchLoadResult | null>(null);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (!cancelled) {
        setResult(loadBatch());
      }
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  if (!result) {
    return <DiscoverLoading />;
  }

  if (result.status !== "ready" || !result.batch) {
    return (
      <EmptyDiscovery
        status={result.status === "ready" ? "missing" : result.status}
      />
    );
  }

  return (
    <DiscoverExperience
      initialBatch={result.batch}
      key={result.batch.savedAt}
    />
  );
}
