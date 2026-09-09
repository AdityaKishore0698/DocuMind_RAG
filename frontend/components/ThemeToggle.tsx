"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/lib/theme-context";
import { IconButton } from "@/components/ui/IconButton";

/**
 * App-bar control that flips between light and dark. The current mode is shown
 * by the icon (sun = currently dark, click for light) and announced via
 * `aria-pressed`. Rendered only inside the authenticated shell, which mounts
 * client-side, so the resolved theme is known on first paint (no hydration
 * guard needed).
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { resolvedTheme, toggle } = useTheme();
  const isDark = resolvedTheme === "dark";

  return (
    <IconButton
      label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-pressed={isDark}
      onClick={toggle}
      className={className}
    >
      {isDark ? <Sun size={20} /> : <Moon size={20} />}
    </IconButton>
  );
}
