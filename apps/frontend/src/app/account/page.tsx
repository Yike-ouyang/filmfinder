"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  AuthError,
  AuthField,
  AuthShell,
  AuthSubmit,
  AuthSuccess,
} from "@/components/auth/AuthShell";
import { useAuth } from "@/components/auth/AuthProvider";
import { errorMessage } from "@/lib/api";

export default function AccountPage() {
  const router = useRouter();
  const { user, loading, signOut, removeAccount } = useAuth();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signOutNow = async () => {
    setBusy(true);
    setError(null);
    try {
      await signOut();
      router.push("/");
    } catch (signOutError) {
      setError(errorMessage(signOutError));
    } finally {
      setBusy(false);
    }
  };

  const deleteNow = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      await removeAccount(password);
      router.push("/");
    } catch (deleteError) {
      setError(errorMessage(deleteError));
    } finally {
      setBusy(false);
    }
  };

  if (loading) {
    return (
      <AuthShell title="Your account" description="Loading your session...">
        <p className="text-sm text-white/50">Checking your session...</p>
      </AuthShell>
    );
  }

  if (!user) {
    return (
      <AuthShell
        title="Sign in required"
        description="Log in to manage your FilmFinder account."
        footer={
          <>
            New here?{" "}
            <Link
              href="/signup"
              className="text-white/75 hover:text-white"
            >
              Create an account
            </Link>
          </>
        }
      >
        <Link
          href="/login"
          className="flex h-11 w-full items-center justify-center rounded-xl bg-white text-sm font-medium text-black transition-opacity hover:opacity-90"
        >
          Log in
        </Link>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title="Your account"
      description="Manage your FilmFinder session and account data."
      footer={
        <Link href="/" className="text-white/75 hover:text-white">
          Back to import
        </Link>
      }
    >
      <div className="space-y-6">
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] px-4 py-3">
          <p className="text-[10px] uppercase tracking-[0.18em] text-white/45">
            Email
          </p>
          <p className="mt-1 break-all text-sm text-white">{user.email}</p>
          <p className="mt-2 text-[11px] text-emerald-200/70">
            {user.email_verified
              ? "Email verified"
              : "Email not verified"}
          </p>
        </div>

        {notice ? <AuthSuccess message={notice} /> : null}
        {error ? <AuthError message={error} /> : null}

        <div className="space-y-3">
          <AuthSubmit type="button" onClick={signOutNow} disabled={busy}>
            {busy ? "Signing out..." : "Log out"}
          </AuthSubmit>
          <Link
            href="/discover"
            className="flex h-11 w-full items-center justify-center rounded-xl border border-white/10 text-sm font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
          >
            Go to Discover
          </Link>
        </div>

        <form
          onSubmit={deleteNow}
          className="space-y-3 border-t border-white/10 pt-6"
        >
          <div>
            <p className="text-sm font-medium text-white">
              Delete my account
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-white/45">
              This permanently deletes your email, imported profile,
              preferences, and recommendation history.
            </p>
          </div>
          <AuthField
            label="Confirm with your password"
            type="password"
            name="delete-password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
              setNotice(null);
            }}
            required
          />
          <button
            type="submit"
            disabled={busy || !password}
            className="h-10 w-full rounded-xl border border-[#e7a6a6]/25 text-xs font-medium text-danger/90 transition-colors hover:bg-[#e7a6a6]/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Delete account permanently
          </button>
        </form>
      </div>
    </AuthShell>
  );
}
