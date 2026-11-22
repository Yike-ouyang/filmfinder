"use client";

import {
  Heart,
  Meh,
  SkipForward,
  ThumbsDown,
  type LucideIcon,
} from "lucide-react";
import { TASTE_ACTIONS } from "@/lib/tasteProfile";
import { cn } from "@/lib/cn";
import type { TasteProfileAction } from "@/types/tasteProfile";

const ICONS: Record<TasteProfileAction, LucideIcon> = {
  loved: Heart,
  okay: Meh,
  disliked: ThumbsDown,
  "no-opinion": SkipForward,
};

interface TasteActionsProps {
  onAction: (action: TasteProfileAction) => void;
  disabled?: boolean;
}

export function TasteActions({
  onAction,
  disabled = false,
}: TasteActionsProps) {
  return (
    <div
      role="group"
      aria-label="Rate this movie"
      className="grid w-full grid-cols-2 gap-2 sm:grid-cols-4 sm:gap-2.5"
    >
      {TASTE_ACTIONS.map((definition) => {
        const Icon = ICONS[definition.action];
        const primary = definition.action === "loved";
        const muted = definition.action === "no-opinion";
        return (
          <button
            key={definition.action}
            type="button"
            onClick={() => onAction(definition.action)}
            disabled={disabled}
            aria-label={`${definition.label} (${definition.shortcut})`}
            title={`${definition.label} (${definition.shortcut})`}
            className={cn(
              "group flex min-h-[74px] flex-col items-center justify-center gap-1.5 rounded-2xl border px-2 py-3 text-center transition-[background-color,border-color,transform] duration-200 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40",
              primary
                ? "border-white bg-white text-black hover:bg-white/90"
                : muted
                  ? "border-white/10 bg-black/35 text-white/55 backdrop-blur-md hover:border-white/20 hover:bg-white/5 hover:text-white/80"
                  : "border-white/15 bg-black/35 text-white/75 backdrop-blur-md hover:border-white/30 hover:bg-white/10 hover:text-white",
            )}
          >
            <Icon
              aria-hidden="true"
              className={cn(
                "h-4 w-4",
                primary && "fill-current",
              )}
            />
            <span className="text-[11px] font-medium leading-tight">
              {definition.label}
            </span>
            <span
              className={cn(
                "font-mono text-[9px] leading-none",
                primary ? "text-black/45" : "text-white/30",
              )}
            >
              {definition.shortcut}
            </span>
          </button>
        );
      })}
    </div>
  );
}
