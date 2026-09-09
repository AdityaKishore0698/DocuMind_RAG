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

/**
 * Theme preference. `"system"` follows the OS via `prefers-color-scheme`;
 * `"light"` / `"dark"` are explicit user choices persisted to localStorage.
 *
 * The actual colour swap is done by toggling `data-theme` on <html>, which the
 * Material 3 token blocks in `app/globals.css` already key off:
 *   - bare `:root`                          → light
 *   - `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])` → dark (system)
 *   - `:root[data-theme="dark"]`            → dark (explicit)
 *
 * A tiny inline script in `app/layout.tsx` applies the stored value before
 * first paint, so there is no flash of the wrong theme on load.
 */
export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "documind-theme";

interface ThemeApi {
  /** The user's preference, including `"system"`. */
  theme: Theme;
  /** What is actually on screen right now — never `"system"`. */
  resolvedTheme: ResolvedTheme;
  setTheme: (next: Theme) => void;
  /** Flip between light and dark (leaves `"system"` once used). */
  toggle: () => void;
}

const ThemeContext = createContext<ThemeApi | null>(null);

function systemPrefersDark(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  );
}

function readStoredTheme(): Theme {
  if (typeof window === "undefined") return "system";
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* private mode / disabled storage */
  }
  return "system";
}

function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.dataset.theme = theme;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Lazy initialisers run on the client during hydration and read the value the
  // inline script already applied; on the server they use the neutral default.
  const [theme, setThemeState] = useState<Theme>(readStoredTheme);
  const [systemDark, setSystemDark] = useState<boolean>(systemPrefersDark);

  // Subscribe to OS scheme changes — only mutates state from the event callback.
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setSystemDark(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const resolvedTheme: ResolvedTheme =
    theme === "system" ? (systemDark ? "dark" : "light") : theme;

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    applyTheme(next);
    try {
      if (next === "system") window.localStorage.removeItem(THEME_STORAGE_KEY);
      else window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  }, []);

  const toggle = useCallback(() => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  }, [resolvedTheme, setTheme]);

  const value = useMemo<ThemeApi>(
    () => ({ theme, resolvedTheme, setTheme, toggle }),
    [theme, resolvedTheme, setTheme, toggle],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeApi {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside <ThemeProvider>");
  return ctx;
}
