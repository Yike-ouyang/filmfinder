"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import {
  AuthError,
  AuthField,
  AuthShell,
  AuthSubmit,
  AuthSuccess,
} from "@/components/auth/AuthShell";
import { errorMessage } from "@/lib/api";
import { resetPassword } from "@/lib/auth";
import { PASSWORD_MIN_LENGTH } from "@/lib/constants";

function ResetPasswordContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    if (!token) {
      setError("This reset link is missing its token.");
      return;
    }
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
      await resetPassword(token, password);
      setDone(true);
    } catch (submitError) {
      setError(errorMessage(submitError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      title={done ? "Password updated" : "Choose a new password"}
      description={
        done
          ? "Your sessions have been signed out. Use your new password to log in."
          : "Pick a strong password for your FilmFinder account."
      }
      footer={
        <>
          {done ? (
            <Link href="/login" className="text-white/75 hover:text-white">
              Continue to log in
            </Link>
          ) : (
            <Link href="/login" className="text-white/75 hover:text-white">
              Back to log in
            </Link>
          )}
        </>
      }
    >
      {done ? (
        <AuthSuccess message="Your password has been changed." />
      ) : (
        <form onSubmit={submit} className="space-y-4" noValidate>
          {!token ? (
            <AuthError message="This reset link is missing its token." />
          ) : null}
          <AuthField
            label="New password"
            type="password"
            name="password"
            autoComplete="new-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            hint={`At least ${PASSWORD_MIN_LENGTH} characters.`}
            required
          />
          <AuthField
            label="Confirm new password"
            type="password"
            name="password-confirmation"
            autoComplete="new-password"
            value={confirmation}
            onChange={(event) => setConfirmation(event.target.value)}
            required
          />
          {error ? <AuthError message={error} /> : null}
          <AuthSubmit type="submit" disabled={submitting || !token}>
            {submitting ? "Updating..." : "Update password"}
          </AuthSubmit>
        </form>
      )}
    </AuthShell>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense
      fallback={
        <AuthShell
          title="Choose a new password"
          description="Loading your reset link..."
        >
          <p className="text-sm text-white/50">Loading...</p>
        </AuthShell>
      }
    >
      <ResetPasswordContent />
    </Suspense>
  );
}
