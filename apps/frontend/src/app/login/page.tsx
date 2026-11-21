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
import { ApiError, errorMessage } from "@/lib/api";
import { resendVerification } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { signIn } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [needsVerification, setNeedsVerification] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resending, setResending] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setNotice(null);
    setNeedsVerification(false);
    setSubmitting(true);
    try {
      await signIn(email, password);
      router.push("/discover");
    } catch (loginError) {
      if (
        loginError instanceof ApiError &&
        loginError.code === "email_not_verified"
      ) {
        setNeedsVerification(true);
      }
      setError(errorMessage(loginError));
    } finally {
      setSubmitting(false);
    }
  };

  const resend = async () => {
    setResending(true);
    setError(null);
    setNotice(null);
    try {
      await resendVerification(email);
      setNotice("A new verification link has been sent.");
    } catch (resendError) {
      setError(errorMessage(resendError));
    } finally {
      setResending(false);
    }
  };

  return (
    <AuthShell
      title="Welcome back"
      description="Sign in to retrieve your saved profile and preferences."
      footer={
        <>
          New to FilmFinder?{" "}
          <Link
            href="/signup"
            className="text-white/75 hover:text-white"
          >
            Create an account
          </Link>
        </>
      }
    >
      <form onSubmit={submit} className="space-y-4" noValidate>
        <AuthField
          label="Email"
          type="email"
          name="email"
          autoComplete="email"
          inputMode="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          required
        />
        <div>
          <AuthField
            label="Password"
            type="password"
            name="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
          <div className="mt-2 text-right">
            <Link
              href="/forgot-password"
              className="text-[11px] text-white/45 transition-colors hover:text-white/75"
            >
              Forgot password?
            </Link>
          </div>
        </div>
        {error ? <AuthError message={error} /> : null}
        {notice ? <AuthSuccess message={notice} /> : null}
        <AuthSubmit type="submit" disabled={submitting}>
          {submitting ? "Signing in..." : "Log in"}
        </AuthSubmit>
        {needsVerification ? (
          <button
            type="button"
            onClick={resend}
            disabled={resending || !email}
            className="h-10 w-full rounded-xl border border-white/10 text-xs font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-40"
          >
            {resending
              ? "Sending..."
              : "Resend verification email"}
          </button>
        ) : null}
      </form>
    </AuthShell>
  );
}
