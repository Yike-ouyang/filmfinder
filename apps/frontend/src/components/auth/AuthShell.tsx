import Link from "next/link";
import type { InputHTMLAttributes } from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/cn";

interface AuthShellProps {
  title: string;
  description: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}

export function AuthShell({
  title,
  description,
  children,
  footer,
}: AuthShellProps) {
  return (
    <main
      id="main"
      className="relative flex min-h-svh items-center justify-center overflow-hidden bg-background px-5 pb-16 pt-28 sm:px-8"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_80%_at_50%_0%,#1b1b1b_0%,#0d0d0d_48%,#080808_82%)]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(65%_45%_at_50%_40%,rgba(255,255,255,0.035),transparent_72%)]"
      />
      <div className="relative z-10 w-full max-w-md">
        <div className="rounded-3xl border border-white/10 bg-black/50 px-6 py-8 shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] backdrop-blur-xl sm:px-9 sm:py-10">
          <Link
            href="/"
            className="text-[10px] font-semibold uppercase tracking-[0.28em] text-white/45 transition-colors hover:text-white"
          >
            FilmFinder
          </Link>
          <h1 className="mt-5 text-3xl font-semibold tracking-[-0.035em] text-white sm:text-4xl">
            {title}
          </h1>
          <p className="mt-3 text-sm leading-relaxed text-white/55">
            {description}
          </p>
          <div className="mt-7">{children}</div>
        </div>
        {footer ? (
          <div className="mt-5 text-center text-xs text-white/45">
            {footer}
          </div>
        ) : null}
      </div>
    </main>
  );
}

interface AuthFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  hint?: string;
}

export function AuthField({ label, hint, className, ...props }: AuthFieldProps) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[10px] uppercase tracking-[0.18em] text-white/50">
        {label}
      </span>
      <input
        {...props}
        className={cn(
          "h-11 w-full rounded-xl border border-white/10 bg-white/[0.03] px-3.5 text-sm text-white outline-none transition-colors placeholder:text-white/30 focus:border-white/30 focus:bg-white/[0.05]",
          className,
        )}
      />
      {hint ? (
        <span className="mt-1.5 block text-[11px] text-white/40">
          {hint}
        </span>
      ) : null}
    </label>
  );
}

export function AuthError({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2.5 rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-3"
    >
      <AlertCircle
        aria-hidden="true"
        className="mt-0.5 h-4 w-4 shrink-0 text-danger"
      />
      <p className="text-xs leading-relaxed text-danger/90">{message}</p>
    </div>
  );
}

export function AuthSuccess({ message }: { message: string }) {
  return (
    <div
      role="status"
      className="flex items-start gap-2.5 rounded-2xl border border-emerald-300/20 bg-emerald-300/[0.06] px-4 py-3"
    >
      <CheckCircle2
        aria-hidden="true"
        className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300"
      />
      <p className="text-xs leading-relaxed text-emerald-100/80">
        {message}
      </p>
    </div>
  );
}

export function AuthSubmit({
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="flex h-11 w-full items-center justify-center rounded-xl bg-white text-sm font-medium text-black transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  );
}
