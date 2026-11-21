"use client";

import { Eye, Heart, RotateCcw, X, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";

interface SwipeControlsProps {
  canUndo: boolean;
  disabled?: boolean;
  onUndo: () => void;
  onPass: () => void;
  onSeen: () => void;
  onInterested: () => void;
}

interface ControlButtonProps {
  label: string;
  accessibleLabel?: string;
  shortcut: string;
  icon: LucideIcon;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  filledIcon?: boolean;
}

function ControlButton({
  label,
  accessibleLabel,
  shortcut,
  icon: Icon,
  onClick,
  disabled = false,
  primary = false,
  filledIcon = false,
}: ControlButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={`${accessibleLabel ?? label} (${shortcut})`}
      title={`${accessibleLabel ?? label} (${shortcut})`}
      className={cn(
        "group flex min-w-14 flex-col items-center gap-2 rounded-2xl px-1 py-1 text-center transition-opacity disabled:pointer-events-none disabled:opacity-25",
      )}
    >
      <span
        className={cn(
          "grid h-12 w-12 place-items-center rounded-full border transition-[background-color,border-color,transform] duration-300 group-hover:scale-105 group-active:scale-95 sm:h-14 sm:w-14",
          primary
            ? "border-white bg-white text-black hover:bg-white/90"
            : "border-white/15 bg-black/35 text-white/70 backdrop-blur-md hover:border-white/30 hover:bg-white/10 hover:text-white",
        )}
      >
        <Icon
          aria-hidden="true"
          className={cn("h-5 w-5", filledIcon && "fill-current")}
        />
      </span>
      <span className="text-[9px] font-medium uppercase tracking-[0.14em] text-white/55 transition-colors group-hover:text-white/80 sm:text-[10px]">
        {label}
      </span>
    </button>
  );
}

export function SwipeControls({
  canUndo,
  disabled = false,
  onUndo,
  onPass,
  onSeen,
  onInterested,
}: SwipeControlsProps) {
  return (
    <div
      role="group"
      aria-label="Movie actions"
      className="flex items-start justify-center gap-1.5 sm:gap-4"
    >
      <ControlButton
        label="Undo"
        shortcut="Backspace"
        icon={RotateCcw}
        onClick={onUndo}
        disabled={!canUndo || disabled}
      />
      <ControlButton
        label="Pass"
        shortcut="Left arrow"
        icon={X}
        onClick={onPass}
        disabled={disabled}
      />
      <ControlButton
        label="Seen"
        accessibleLabel="Already seen"
        shortcut="Down arrow"
        icon={Eye}
        onClick={onSeen}
        disabled={disabled}
      />
      <ControlButton
        label="Interested"
        shortcut="Right arrow"
        icon={Heart}
        onClick={onInterested}
        disabled={disabled}
        primary
        filledIcon
      />
    </div>
  );
}
