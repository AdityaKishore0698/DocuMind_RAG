"use client";

import { useRef, type ClipboardEvent, type KeyboardEvent } from "react";
import { cn } from "@/lib/cn";

interface Props {
  value: string;
  onChange: (next: string) => void;
  length?: number;
  disabled?: boolean;
  "aria-label"?: string;
}

/** Segmented numeric one-time-code input. Paste-aware, arrow/backspace-aware. */
export function OtpInput({
  value,
  onChange,
  length = 6,
  disabled,
  "aria-label": ariaLabel = "One-time code",
}: Props) {
  const refs = useRef<Array<HTMLInputElement | null>>([]);
  const digits = value.split("").slice(0, length);

  function set(i: number, char: string) {
    const next = value.split("");
    next[i] = char;
    onChange(next.join("").replace(/\D/g, "").slice(0, length));
  }

  function onKeyDown(i: number, e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Backspace" && !digits[i] && i > 0) {
      refs.current[i - 1]?.focus();
    } else if (e.key === "ArrowLeft" && i > 0) {
      refs.current[i - 1]?.focus();
    } else if (e.key === "ArrowRight" && i < length - 1) {
      refs.current[i + 1]?.focus();
    }
  }

  function onPaste(e: ClipboardEvent<HTMLInputElement>) {
    e.preventDefault();
    const text = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, length);
    if (text) {
      onChange(text);
      refs.current[Math.min(text.length, length - 1)]?.focus();
    }
  }

  return (
    <div className="flex justify-between gap-2" role="group" aria-label={ariaLabel}>
      {Array.from({ length }).map((_, i) => (
        <input
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          inputMode="numeric"
          autoComplete={i === 0 ? "one-time-code" : "off"}
          maxLength={1}
          disabled={disabled}
          aria-label={`Digit ${i + 1}`}
          value={digits[i] ?? ""}
          onChange={(e) => {
            const char = e.target.value.replace(/\D/g, "").slice(-1);
            if (char) {
              set(i, char);
              refs.current[Math.min(i + 1, length - 1)]?.focus();
            } else {
              set(i, "");
            }
          }}
          onKeyDown={(e) => onKeyDown(i, e)}
          onPaste={onPaste}
          className={cn(
            "h-14 w-full min-w-0 rounded-md border bg-md-surface-container-highest text-center text-xl font-medium text-md-on-surface",
            "border-md-outline outline-none transition-colors",
            "focus:border-md-primary focus:ring-2 focus:ring-md-primary/30",
            "disabled:opacity-40",
          )}
        />
      ))}
    </div>
  );
}
