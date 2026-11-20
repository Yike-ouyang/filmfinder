"use client";

import Image from "next/image";
import { useState, type ReactNode } from "react";
import { cn } from "@/lib/cn";
import { getTmdbBackdropUrl, getTmdbPosterUrl } from "@/lib/tmdb";

interface TmdbMediaProps {
  path: string | null | undefined;
  variant: "poster" | "backdrop";
  alt: string;
  sizes: string;
  className?: string;
  imageClassName?: string;
  preload?: boolean;
  fallback: ReactNode;
}

export function TmdbMedia({
  path,
  variant,
  alt,
  sizes,
  className,
  imageClassName,
  preload = false,
  fallback,
}: TmdbMediaProps) {
  const src =
    variant === "poster" ? getTmdbPosterUrl(path) : getTmdbBackdropUrl(path);
  const [status, setStatus] = useState<{
    src: string | null;
    loaded: boolean;
    failed: boolean;
  }>({ src, loaded: false, failed: false });
  const current = status.src === src;
  const loaded = current && status.loaded;
  const failed = current && status.failed;

  if (!src || failed) {
    return <div className={cn("h-full w-full", className)}>{fallback}</div>;
  }

  return (
    <div className={cn("relative h-full w-full overflow-hidden", className)}>
      {!loaded ? (
        <div
          aria-hidden="true"
          className="absolute inset-0 animate-pulse bg-[linear-gradient(140deg,#161616,#0b0b0b_60%,#181818)]"
        />
      ) : null}
      <Image
        src={src}
        alt={alt}
        fill
        sizes={sizes}
        preload={preload}
        loading={preload ? undefined : "lazy"}
        quality={85}
        draggable={false}
        onLoad={() => setStatus({ src, loaded: true, failed: false })}
        onError={() => setStatus({ src, loaded: false, failed: true })}
        className={cn(
          "object-cover transition-[opacity,transform] duration-700 ease-out",
          loaded ? "opacity-100" : "opacity-0",
          imageClassName,
        )}
      />
    </div>
  );
}
