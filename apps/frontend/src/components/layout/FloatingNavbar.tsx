"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";
import { SITE_NAME } from "@/lib/constants";

const links = [
  { href: "/", label: "Home" },
  { href: "/discover", label: "Discover" },
];

export function FloatingNavbar() {
  const pathname = usePathname();

  return (
    <header className="safe-top pointer-events-none fixed inset-x-0 top-0 z-50 flex justify-center px-4">
      <nav
        aria-label="Main navigation"
        className="pointer-events-auto flex items-center gap-1 rounded-full border border-white/10 bg-black/55 p-1.5 pl-4 shadow-[0_16px_50px_-20px_rgba(0,0,0,0.9)] backdrop-blur-xl"
      >
        <Link
          href="/"
          className="mr-2 flex items-center gap-2 text-[11px] font-semibold tracking-[0.22em] text-white transition-opacity hover:opacity-70 sm:mr-4 sm:text-xs"
          aria-label={`${SITE_NAME} home`}
        >
          <span
            aria-hidden="true"
            className="h-1.5 w-1.5 rounded-full bg-white"
          />
          {SITE_NAME.toUpperCase()}
        </Link>
        <div className="flex items-center gap-0.5">
          {links.map((link) => {
            const active =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "rounded-full px-3 py-2 text-xs font-medium transition-colors duration-300 sm:px-4 sm:text-[13px]",
                  active
                    ? "bg-white text-black"
                    : "text-white/60 hover:bg-white/5 hover:text-white",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
