"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { cn } from "@/lib/cn";
import { SITE_NAME } from "@/lib/constants";

const links = [
  { href: "/", label: "Home" },
  { href: "/discover", label: "Discover" },
  { href: "/lists", label: "My Lists" },
];

const navItemClass =
  "rounded-full px-3 py-2 text-xs font-medium transition-colors duration-300 sm:px-4 sm:text-[13px]";

export function FloatingNavbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, signOut } = useAuth();
  const [signingOut, setSigningOut] = useState(false);

  const handleLogout = async () => {
    setSigningOut(true);
    try {
      await signOut();
      router.push("/");
    } catch {
      router.push("/");
    } finally {
      setSigningOut(false);
    }
  };

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
                  navItemClass,
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
        <div className="ml-1 flex items-center gap-0.5 border-l border-white/10 pl-1.5">
          {loading ? (
            <span className="px-3 py-2 text-xs text-white/25">
              ...
            </span>
          ) : user ? (
            <>
              <Link
                href="/account"
                aria-current={pathname.startsWith("/account") ? "page" : undefined}
                className={cn(
                  navItemClass,
                  pathname.startsWith("/account")
                    ? "bg-white text-black"
                    : "text-white/60 hover:bg-white/5 hover:text-white",
                )}
              >
                Account
              </Link>
              <button
                type="button"
                onClick={handleLogout}
                disabled={signingOut}
                className={cn(
                  navItemClass,
                  "text-white/45 hover:bg-white/5 hover:text-white disabled:opacity-40",
                )}
              >
                {signingOut ? "..." : "Log out"}
              </button>
            </>
          ) : (
            <Link
              href="/login"
              aria-current={pathname.startsWith("/login") ? "page" : undefined}
              className={cn(
                navItemClass,
                pathname.startsWith("/login")
                  ? "bg-white text-black"
                  : "text-white/60 hover:bg-white/5 hover:text-white",
              )}
            >
              Log in
            </Link>
          )}
        </div>
      </nav>
    </header>
  );
}
