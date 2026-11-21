"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import {
  AuthError,
  AuthShell,
  AuthSuccess,
} from "@/components/auth/AuthShell";
import { errorMessage } from "@/lib/api";
import { verifyEmail } from "@/lib/auth";

type VerifyState = "pending" | "success" | "error";

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const [state, setState] = useState<VerifyState>("pending");
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (started.current || !token) {
      return;
    }
    started.current = true;
    verifyEmail(token)
      .then(() => setState("success"))
      .catch((verifyError) => {
        setState("error");
        setError(errorMessage(verifyError));
      });
  }, [token]);

  const displayState: VerifyState = token ? state : "error";
  const displayError = token
    ? error
    : "This verification link is missing its token.";

  return (
    <AuthShell
      title={
        displayState === "success"
          ? "Email verified"
          : displayState === "error"
            ? "Verification failed"
            : "Verifying your email"
      }
      description={
        displayState === "success"
          ? "Your account is active. You can now sign in and import your Letterboxd profile."
          : displayState === "error"
            ? "The link may have expired or already been used."
            : "This only takes a moment."
      }
      footer={
        <>
          {displayState === "success" ? (
            <Link href="/login" className="text-white/75 hover:text-white">
              Continue to log in
            </Link>
          ) : (
            <Link href="/signup" className="text-white/75 hover:text-white">
              Create a new account
            </Link>
          )}
        </>
      }
    >
      <div className="space-y-4">
        {displayState === "pending" ? (
          <p className="text-sm text-white/50">Checking your link...</p>
        ) : null}
        {displayState === "success" ? (
          <AuthSuccess message="Your email address is now confirmed." />
        ) : null}
        {displayState === "error" && displayError ? (
          <AuthError message={displayError} />
        ) : null}
        {displayState === "error" ? (
          <p className="text-center text-[11px] leading-relaxed text-white/40">
            Need a new link? Sign in with your email and password, then
            request a new verification email.
          </p>
        ) : null}
      </div>
    </AuthShell>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense
      fallback={
        <AuthShell
          title="Verifying your email"
          description="This only takes a moment."
        >
          <p className="text-sm text-white/50">Checking your link...</p>
        </AuthShell>
      }
    >
      <VerifyEmailContent />
    </Suspense>
  );
}
