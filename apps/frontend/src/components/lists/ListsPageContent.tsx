"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Film, LoaderCircle, RefreshCw } from "lucide-react";
import {
  AuthError,
  AuthShell,
} from "@/components/auth/AuthShell";
import { useAuth } from "@/components/auth/AuthProvider";
import { SwipePosterCard } from "@/components/lists/SwipePosterCard";
import { errorMessage } from "@/lib/api";
import { fetchSwipeLists, removeSwipe } from "@/lib/swipes";
import { cn } from "@/lib/cn";
import { TmdbAttribution } from "@/components/ui/TmdbAttribution";
import type { SwipeItem, SwipeLists } from "@/types/swipes";

type ListTab = "interested" | "not_interested";

const TABS: { key: ListTab; label: string }[] = [
  { key: "interested", label: "Interested" },
  { key: "not_interested", label: "Not Interested" },
];

function ListsLoading() {
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
        aria-label="Loading your lists"
        className="relative z-10 flex flex-col items-center gap-4 text-white/55"
      >
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin text-white/60"
        />
        <p className="text-[10px] uppercase tracking-[0.28em]">
          Loading your lists
        </p>
      </div>
    </main>
  );
}

export function ListsPageContent() {
  const { user, loading: authLoading } = useAuth();
  const [lists, setLists] = useState<SwipeLists | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ListTab>("interested");
  const [removingId, setRemovingId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setLists(await fetchSwipeLists());
    } catch (loadError) {
      setError(errorMessage(loadError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!user) {
      return;
    }
    const timer = window.setTimeout(() => {
      void load();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [user, load]);

  const handleRemove = async (item: SwipeItem) => {
    if (!lists || removingId !== null) {
      return;
    }
    const key: ListTab =
      item.action === "interested" ? "interested" : "not_interested";
    const previous = lists;
    setRemovingId(item.internal_movie_id);
    setError(null);
    setLists({
      ...lists,
      [key]: lists[key].filter(
        (entry) => entry.internal_movie_id !== item.internal_movie_id,
      ),
      counts: {
        ...lists.counts,
        [key]: Math.max(0, lists.counts[key] - 1),
      },
    });
    try {
      await removeSwipe(item.internal_movie_id);
    } catch (removeError) {
      setLists(previous);
      setError(errorMessage(removeError));
    } finally {
      setRemovingId(null);
    }
  };

  if (authLoading) {
    return <ListsLoading />;
  }

  if (!user) {
    return (
      <AuthShell
        title="My Lists"
        description="Create an account to save the films you are interested in and the ones you want to skip."
        footer={
          <>
            Questions?{" "}
            <Link href="/" className="text-white/75 hover:text-white">
              Back to FilmFinder
            </Link>
          </>
        }
      >
        <div className="space-y-3">
          <Link
            href="/signup"
            className="flex h-11 w-full items-center justify-center rounded-xl bg-white text-sm font-medium text-black transition-opacity hover:opacity-90"
          >
            Create an account
          </Link>
          <Link
            href="/login"
            className="flex h-11 w-full items-center justify-center rounded-xl border border-white/10 text-sm font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
          >
            Log in
          </Link>
        </div>
      </AuthShell>
    );
  }

  const items = lists?.[activeTab] ?? [];
  const counts = lists?.counts ?? { interested: 0, not_interested: 0 };

  return (
    <main
      id="main"
      className="relative min-h-svh overflow-hidden bg-background px-5 pb-20 pt-28 sm:px-8"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_70%_at_50%_0%,#1a1a1a_0%,#0d0d0d_46%,#080808_82%)]"
      />
      <div className="relative z-10 mx-auto w-full max-w-6xl">
        <p className="text-[10px] font-medium uppercase tracking-[0.32em] text-white/45">
          Saved preferences
        </p>
        <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em] text-white sm:text-5xl">
          My Lists
        </h1>
        <p className="mt-3 max-w-xl text-sm leading-relaxed text-white/50">
          The films you saved while discovering. Changes are stored in your
          account.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <div
            role="tablist"
            aria-label="My Lists categories"
            className="flex items-center gap-1 rounded-full border border-white/10 bg-black/40 p-1"
          >
            {TABS.map((tab) => {
              const active = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  type="button"
                  role="tab"
                  id={`tab-${tab.key}`}
                  aria-selected={active}
                  aria-controls={`panel-${tab.key}`}
                  onClick={() => setActiveTab(tab.key)}
                  className={cn(
                    "rounded-full px-4 py-2 text-xs font-medium transition-colors",
                    active
                      ? "bg-white text-black"
                      : "text-white/55 hover:bg-white/5 hover:text-white",
                  )}
                >
                  {tab.label}
                  <span
                    className={cn(
                      "ml-2 text-[10px]",
                      active ? "text-black/50" : "text-white/35",
                    )}
                  >
                    {counts[tab.key]}
                  </span>
                </button>
              );
            })}
          </div>
          <button
            type="button"
            onClick={() => void load()}
            disabled={loading}
            className="inline-flex h-9 items-center gap-2 rounded-full border border-white/10 px-3 text-[11px] font-medium text-white/55 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-40"
          >
            <RefreshCw
              aria-hidden="true"
              className={cn("h-3.5 w-3.5", loading && "animate-spin")}
            />
            Refresh
          </button>
        </div>

        {error ? (
          <div className="mt-6 max-w-xl">
            <AuthError message={error} />
          </div>
        ) : null}

        <div
          role="tabpanel"
          id={`panel-${activeTab}`}
          aria-labelledby={`tab-${activeTab}`}
          className="mt-8"
        >
          {loading && !lists ? (
            <div className="flex items-center gap-3 text-sm text-white/50">
              <LoaderCircle
                aria-hidden="true"
                className="h-4 w-4 animate-spin"
              />
              Loading your {activeTab === "interested" ? "interested" : "skipped"} films...
            </div>
          ) : items.length === 0 ? (
            <div className="rounded-3xl border border-white/10 bg-white/[0.02] px-6 py-14 text-center">
              <Film
                aria-hidden="true"
                className="mx-auto h-6 w-6 text-white/30"
              />
              <p className="mt-4 text-sm font-medium text-white/70">
                No films here yet.
              </p>
              <p className="mt-2 text-xs text-white/40">
                Swipe through Discover to build this list.
              </p>
              <Link
                href="/discover"
                className="mt-6 inline-flex h-10 items-center justify-center rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-90"
              >
                Go to Discover
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {items.map((item) => (
                <SwipePosterCard
                  key={item.internal_movie_id}
                  item={item}
                  removing={removingId === item.internal_movie_id}
                  onRemove={handleRemove}
                />
              ))}
            </div>
          )}
        </div>

        <footer className="mt-16 text-center">
          <TmdbAttribution />
        </footer>
      </div>
    </main>
  );
}
