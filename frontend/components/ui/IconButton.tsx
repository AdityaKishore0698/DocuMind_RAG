"use client";

import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/cn";
import { Spinner } from "./Spinner";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Required — icon buttons have no visible text. */
  label: string;
  loading?: boolean;
  variant?: "standard" | "filled" | "tonal";
}

const VARIANTS = {
  standard: "text-md-on-surface-variant hover:bg-md-on-surface/[0.08]",
  filled: "bg-md-primary text-md-on-primary hover:elev-1",
  tonal: "bg-md-secondary-container text-md-on-secondary-container hover:elev-1",
};

export const IconButton = forwardRef<HTMLButtonElement, Props>(function IconButton(
  { label, loading, variant = "standard", disabled, className, children, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled || loading}
      className={cn(
        "state inline-grid h-11 w-11 place-items-center rounded-full transition-colors duration-150",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-md-primary",
        "disabled:pointer-events-none disabled:opacity-40",
        VARIANTS[variant],
        className,
      )}
      {...props}
    >
      {loading ? <Spinner size={18} /> : children}
    </button>
  );
});
