"use client";

import Link from "next/link";
import { useState } from "react";
import {
  AuthError,
  AuthField,
  AuthShell,
  AuthSubmit,
  AuthSuccess,
} from "@/components/auth/AuthShell";
import { errorMessage } from "@/lib/api";
import { resendVerification, signup } from "@/lib/auth";
import { PASSWORD_MIN_LENGTH } from "@/lib/constants";
import type { SignupResponse } from "@/types/auth";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SignupResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resending, setResending] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setNotice(null);
    if (password.length < PASSWORD_MIN_LENGTH) {
      setError(
        `Use a password with at least ${PASSWORD_MIN_LENGTH} characters.`,
      );
      return;
    }
    if (password !== confirmation) {
      setError("The passwords do not match.");
      return;
    }
    setSubmitting(true);
    try {
      setResult(await signup(email, password));
    } catch (submitError) {
      setError(errorMessage(submitError));
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

  if (result) {
    const verificationRequired = result.verification_required;
    return (
      <AuthShell
        title={verificationRequired ? "Check your inbox" : "Account created"}
        description={
          verificationRequired
            ? "We sent a verification link to your email address. Open it to activate your account."
            : "Your account is ready. Log in with your email address and password."
        }
        footer={
          <>
            {verificationRequired ? "Already verified? " : "Ready to start? "}
            <Link href="/login" className="text-white/75 hover:text-white">
              Log in
            </Link>
          </>
        }
      >
        <div className="space-y-4">
          <AuthSuccess message={result.message} />
          {verificationRequired ? (
            <>
              {notice ? <AuthSuccess message={notice} /> : null}
              {error ? <AuthError message={error} /> : null}
              <AuthSubmit
                type="button"
                onClick={resend}
                disabled={resending}
              >
                {resending ? "Sending..." : "Resend verification email"}
              </AuthSubmit>
              <p className="text-center text-[11px] leading-relaxed text-white/40">
                The verification link expires after 24 hours. Check your spam
                folder if it does not arrive.
              </p>
            </>
          ) : (
            <Link
              href="/login"
              className="flex h-11 w-full items-center justify-center rounded-xl bg-white text-sm font-medium text-black transition-opacity hover:opacity-90"
            >
              Log in
            </Link>
          )}
        </div>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title="Create your account"
      description="Save your Letterboxd profile, preferences, and recommendations across visits."
      footer={
        <>
          Already have an account?{" "}
          <Link
            href="/login"
            className="text-white/75 hover:text-white"
          >
            Log in
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
        <AuthField
          label="Password"
          type="password"
          name="password"
          autoComplete="new-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          hint={`At least ${PASSWORD_MIN_LENGTH} characters.`}
          required
        />
        <AuthField
          label="Confirm password"
          type="password"
          name="password-confirmation"
          autoComplete="new-password"
          value={confirmation}
          onChange={(event) => setConfirmation(event.target.value)}
          required
        />
        {error ? <AuthError message={error} /> : null}
        <AuthSubmit type="submit" disabled={submitting}>
          {submitting ? "Creating account..." : "Create account"}
        </AuthSubmit>
      </form>
      <p className="mt-5 text-center text-[11px] leading-relaxed text-white/40">
        We will never store your Letterboxd archive. Only your imported
        ratings and preferences are saved.
      </p>
    </AuthShell>
  );
}
