"use client";

import { useState, type FormEvent } from "react";
import { ArrowLeft, Eye, EyeOff } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { IconButton } from "@/components/ui/IconButton";
import { TextField } from "@/components/ui/TextField";

type Mode = "signin" | "register" | "reset";

export default function LoginView() {
  const {
    emailConfirmed,
    signInWithPassword,
    signUpWithPassword,
    signInWithGoogle,
    sendPasswordReset,
  } = useAuth();

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [busy, setBusy] = useState<null | "form" | "google">(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  function go(next: Mode) {
    setMode(next);
    setError(null);
    setNotice(null);
  }

  async function run(kind: "form" | "google", fn: () => Promise<void>) {
    setError(null);
    setBusy(kind);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(null);
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (mode === "signin") {
      await run("form", () => signInWithPassword(email.trim(), password));
    } else if (mode === "register") {
      await run("form", async () => {
        const { needsConfirmation } = await signUpWithPassword(
          email.trim(),
          password,
        );
        setNotice(
          needsConfirmation
            ? "Account created. Please check your email and click the confirmation link to verify your account."
            : "Account created — signing you in…",
        );
      });
    } else {
      await run("form", async () => {
        await sendPasswordReset(email.trim());
        setNotice("Check your email for a password-reset link.");
      });
    }
  }

  const isAuthTab = mode === "signin" || mode === "register";
  const shownNotice =
    notice ??
    (emailConfirmed && mode === "signin"
      ? "Your email is confirmed. Please sign in to continue."
      : null);

  return (
    <main className="flex min-h-dvh items-center justify-center bg-md-surface p-4">
      <Card className="w-full max-w-[26rem] p-6 sm:p-9">
        <header className="mb-7 text-center">
          <h1 className="t-headline-m">DocuMind</h1>
          <p className="t-body-m mt-1 text-md-on-surface-variant">
            Chat with your documents.
          </p>
        </header>

        {isAuthTab && (
          <div
            role="tablist"
            aria-label="Authentication mode"
            className="mb-7 grid grid-cols-2 rounded-full bg-md-surface-container-high p-1 t-label-l"
          >
            {(["signin", "register"] as const).map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={mode === m}
                type="button"
                onClick={() => go(m)}
                className={`rounded-full px-4 py-2 transition-colors ${
                  mode === m
                    ? "bg-md-primary text-md-on-primary"
                    : "text-md-on-surface-variant hover:text-md-on-surface"
                }`}
              >
                {m === "signin" ? "Sign in" : "Register"}
              </button>
            ))}
          </div>
        )}

        {mode === "reset" && (
          <button
            type="button"
            onClick={() => go("signin")}
            className="mb-5 inline-flex items-center gap-1 t-label-l text-md-on-surface-variant hover:text-md-on-surface"
          >
            <ArrowLeft size={16} /> Back to sign in
          </button>
        )}

        <form onSubmit={onSubmit} className="space-y-5" noValidate>
          <TextField
            label="Email address"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          {isAuthTab && (
            <TextField
              label="Password"
              type={showPw ? "text" : "password"}
              required
              minLength={8}
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              helperText={mode === "register" ? "At least 8 characters." : undefined}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              trailing={
                <IconButton
                  label={showPw ? "Hide password" : "Show password"}
                  onClick={() => setShowPw((v) => !v)}
                  tabIndex={-1}
                >
                  {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
                </IconButton>
              }
            />
          )}

          {error && (
            <p
              role="alert"
              className="rounded-lg bg-md-error-container px-3 py-2 t-body-m text-md-on-error-container"
            >
              {error}
            </p>
          )}
          {shownNotice && !error && (
            <p role="status" className="t-body-m text-md-on-surface-variant">
              {shownNotice}
            </p>
          )}

          <Button type="submit" size="lg" fullWidth loading={busy === "form"}>
            {mode === "signin"
              ? "Sign in"
              : mode === "register"
                ? "Create account"
                : "Send reset link"}
          </Button>
        </form>

        {isAuthTab && (
          <>
            <div className="my-6 flex items-center gap-3 t-label-m text-md-on-surface-variant">
              <span className="h-px flex-1 bg-md-outline-variant" />
              OR
              <span className="h-px flex-1 bg-md-outline-variant" />
            </div>
            <Button
              variant="outlined"
              size="lg"
              fullWidth
              loading={busy === "google"}
              leadingIcon={<GoogleMark />}
              onClick={() => run("google", signInWithGoogle)}
            >
              Continue with Google
            </Button>
          </>
        )}

        {mode === "signin" && (
          <button
            type="button"
            onClick={() => go("reset")}
            className="mt-5 block w-full text-center t-label-l text-md-primary hover:underline"
          >
            Forgot your password?
          </button>
        )}
      </Card>
    </main>
  );
}

function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.6l6.8-6.8C35.6 2.4 30.2 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.9 6.1C12.4 13.3 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.5 3-2.2 5.5-4.7 7.2l7.3 5.7c4.3-3.9 6.9-9.7 6.9-17.4z" />
      <path fill="#FBBC05" d="M10.5 28.3a14.6 14.6 0 0 1 0-8.6l-7.9-6.1a24 24 0 0 0 0 20.8l7.9-6.1z" />
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.3-5.7c-2 1.4-4.7 2.3-8.6 2.3-6.3 0-11.6-3.8-13.5-9.3l-7.9 6.1C6.5 42.6 14.6 48 24 48z" />
    </svg>
  );
}
