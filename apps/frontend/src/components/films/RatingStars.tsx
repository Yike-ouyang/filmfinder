"use client";

import { Star } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/cn";

interface RatingStarsProps {
  value: number | null;
  onChange?: (value: number) => void;
  readOnly?: boolean;
  disabled?: boolean;
  size?: "sm" | "md" | "lg";
  label?: string;
  className?: string;
}

const SIZE_CLASSES = {
  sm: "h-3.5 w-3.5",
  md: "h-5 w-5",
  lg: "h-6 w-6",
} as const;

function filledPercent(value: number | null, index: number): number {
  if (value === null || value === undefined) {
    return 0;
  }
  const stars = Math.max(0, Math.min(5, value / 2));
  return Math.max(0, Math.min(100, (stars - index) * 100));
}

export function RatingStars({
  value,
  onChange,
  readOnly = false,
  disabled = false,
  size = "md",
  label = "Rating",
  className,
}: RatingStarsProps) {
  const [hover, setHover] = useState<number | null>(null);
  const display = hover ?? value;
  const starClass = SIZE_CLASSES[size];

  return (
    <div
      className={cn("relative inline-flex flex-col", className)}
      role={readOnly ? undefined : "radiogroup"}
      aria-label={label}
    >
      <div className="flex items-center" aria-hidden="true">
        {Array.from({ length: 5 }, (_, index) => {
          const percent = filledPercent(display, index);
          return (
            <span key={index} className="relative block">
              <Star
                className={cn(starClass, "text-white/15")}
                strokeWidth={1.5}
              />
              <span
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${percent}%` }}
              >
                <Star
                  className={cn(
                    starClass,
                    "fill-[#f5c518] text-[#f5c518]",
                  )}
                  strokeWidth={1.5}
                />
              </span>
            </span>
          );
        })}
      </div>
      {readOnly ? null : (
        <div className="absolute inset-0 flex" onMouseLeave={() => setHover(null)}>
          {Array.from({ length: 10 }, (_, index) => {
            const rating = index + 1;
            return (
              <button
                key={rating}
                type="button"
                disabled={disabled}
                role="radio"
                aria-checked={value === rating}
                aria-label={`Rate ${rating / 2} stars out of 5 (${rating}/10)`}
                onMouseEnter={() => setHover(rating)}
                onFocus={() => setHover(rating)}
                onBlur={() => setHover(null)}
                onClick={() => onChange?.(rating)}
                className="h-full flex-1 rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed"
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
