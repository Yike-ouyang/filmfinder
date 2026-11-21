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
import { forgotPassword } from "@/lib/auth";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await forgotPassword(email);
      setSent(true);
    } catch (submitError) {
      setError(errorMessage(submitError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      title="Reset your password"
      description="Enter your email address and we will send reset instructions if an account exists."
      footer={
        <>
          Remembered it?{" "}
          <Link href="/login" className="text-white/75 hover:text-white">
            Back to log in
          </Link>
        </>
      }
    >
      {sent ? (
        <div className="space-y-4">
          <AuthSuccess message="If an account exists for this email, reset instructions have been sent." />
          <p className="text-center text-[11px] leading-relaxed text-white/40">
            Check your spam folder if the message does not arrive within a
            few minutes.
          </p>
        </div>
      ) : (
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
          {error ? <AuthError message={error} /> : null}
          <AuthSubmit type="submit" disabled={submitting}>
            {submitting ? "Sending..." : "Send reset link"}
          </AuthSubmit>
        </form>
      )}
    </AuthShell>
  );
}
