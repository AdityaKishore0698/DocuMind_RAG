"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "./supabase";

/** Result of a signup attempt. `needsConfirmation` is true when Supabase has
 *  sent a confirmation email and withheld the session until the user clicks the
 *  link; false only when the project has email confirmation disabled. */
export interface SignUpResult {
  needsConfirmation: boolean;
}

/** sessionStorage flag that survives the full-page redirect to Google and back,
 *  letting us tell an OAuth return apart from a confirmation-link landing. */
const OAUTH_FLOW_FLAG = "documind-oauth-flow";

export interface Auth {
  session: Session | null;
  ready: boolean;
  /** True when the user just arrived from an email-confirmation link. The UI
   *  uses this to show a "please sign in" notice on the login screen. */
  emailConfirmed: boolean;
  signUpWithPassword: (email: string, password: string) => Promise<SignUpResult>;
  signInWithPassword: (email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  sendPasswordReset: (email: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<Auth | null>(null);

function fail(error: { message: string } | null) {
  if (error) throw new Error(error.message);
}

function setFlag(key: string) {
  try {
    window.sessionStorage.setItem(key, "1");
  } catch {
    /* ignore */
  }
}

/**
 * How the current page was loaded, w.r.t. auth. Memoised at module scope so it
 * is computed exactly once per real page load and stays stable across React
 * Strict Mode's double-mounting (which would otherwise re-consume the flag).
 *
 *  - `oauth`  — return from the Google OAuth redirect we initiated
 *  - `link`   — landed on an email link (confirmation / recovery / magic)
 *  - `normal` — a plain visit
 */
type PageLoadKind = "oauth" | "link" | "normal";
let cachedPageLoadKind: PageLoadKind | null = null;

function classifyPageLoad(): PageLoadKind {
  if (cachedPageLoadKind) return cachedPageLoadKind;
  const loc = window.location;
  const linkReturn =
    new URLSearchParams(loc.search).has("code") ||
    loc.hash.includes("access_token") ||
    loc.hash.includes("error_description");
  let oauth = false;
  try {
    oauth = window.sessionStorage.getItem(OAUTH_FLOW_FLAG) === "1";
    window.sessionStorage.removeItem(OAUTH_FLOW_FLAG);
  } catch {
    /* ignore */
  }
  cachedPageLoadKind = !linkReturn ? "normal" : oauth ? "oauth" : "link";
  return cachedPageLoadKind;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [emailConfirmed, setEmailConfirmed] = useState(false);

  useEffect(() => {
    let active = true;

    // A confirmation link (or any non-OAuth auth link) lands with `?code=` /
    // `#access_token`; Google OAuth returns the same way but is flagged before
    // we redirect. We want a confirmation link to drop the user on the sign-in
    // screen, not log them straight in.
    const emailConfirmReturn = classifyPageLoad() === "link";

    const cleanUrl = () =>
      window.history.replaceState({}, "", window.location.pathname);

    supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      if (emailConfirmReturn) {
        // Let onAuthStateChange classify a session if one is mid-exchange.
        // If nothing is pending, the address was still verified server-side —
        // show the sign-in screen with a confirmation notice.
        if (!data.session) {
          setEmailConfirmed(true);
          cleanUrl();
          setReady(true);
        }
        return;
      }
      setSession(data.session);
      setReady(true);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((event, next) => {
      if (!active) return;
      if (emailConfirmReturn && event === "SIGNED_IN") {
        // Confirmation link established a session — tear it down so the user
        // signs in manually.
        void supabase.auth.signOut();
        setEmailConfirmed(true);
        cleanUrl();
        return;
      }
      setSession(next);
      setReady(true);
    });

    return () => {
      active = false;
      sub.subscription.unsubscribe();
    };
  }, []);

  const signUpWithPassword = useCallback(
    async (email: string, password: string): Promise<SignUpResult> => {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { emailRedirectTo: window.location.origin },
      });
      fail(error);
      // Supabase returns a user with an empty `identities` array (and no error)
      // when the email already belongs to a confirmed account — enumeration
      // protection. Surface that as a normal "please sign in" message.
      if (data.user && (data.user.identities?.length ?? 0) === 0) {
        throw new Error(
          "This email is already registered. Try signing in instead.",
        );
      }
      // A session here means email confirmation is disabled for the project;
      // onAuthStateChange will pick it up and the app continues.
      return { needsConfirmation: !data.session };
    },
    [],
  );

  const signInWithPassword = useCallback(async (email: string, password: string) => {
    fail((await supabase.auth.signInWithPassword({ email, password })).error);
  }, []);

  const signInWithGoogle = useCallback(async () => {
    setFlag(OAUTH_FLOW_FLAG);
    fail(
      (
        await supabase.auth.signInWithOAuth({
          provider: "google",
          options: { redirectTo: window.location.origin },
        })
      ).error,
    );
  }, []);

  const sendPasswordReset = useCallback(async (email: string) => {
    fail(
      (
        await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: window.location.origin,
        })
      ).error,
    );
  }, []);

  const signOut = useCallback(async () => {
    setEmailConfirmed(false);
    await supabase.auth.signOut();
  }, []);

  const value = useMemo<Auth>(
    () => ({
      session,
      ready,
      emailConfirmed,
      signUpWithPassword,
      signInWithPassword,
      signInWithGoogle,
      sendPasswordReset,
      signOut,
    }),
    [
      session,
      ready,
      emailConfirmed,
      signUpWithPassword,
      signInWithPassword,
      signInWithGoogle,
      sendPasswordReset,
      signOut,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): Auth {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

/** The current Supabase access token (for the API client), or null. */
export function useAccessToken(): string | null {
  return useAuth().session?.access_token ?? null;
}
