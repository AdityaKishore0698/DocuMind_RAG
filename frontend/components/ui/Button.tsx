"use client";

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/cn";
import { Spinner } from "./Spinner";

type Variant = "filled" | "tonal" | "outlined" | "text";
type Size = "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  filled: "bg-md-primary text-md-on-primary hover:elev-1",
  tonal: "bg-md-secondary-container text-md-on-secondary-container hover:elev-1",
  outlined:
    "border border-md-outline text-md-primary bg-transparent hover:bg-md-primary/[0.06]",
  text: "text-md-primary bg-transparent hover:bg-md-primary/[0.08]",
};

const SIZES: Record<Size, string> = {
  md: "h-10 px-5 text-sm",
  lg: "h-12 px-6 text-[0.95rem]",
};

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  fullWidth?: boolean;
  leadingIcon?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  {
    variant = "filled",
    size = "md",
    loading = false,
    fullWidth = false,
    leadingIcon,
    disabled,
    className,
    children,
    ...props
  },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        "state inline-flex select-none items-center justify-center gap-2 rounded-full font-medium tracking-wide",
        "transition-[box-shadow,background-color,opacity] duration-150 ease-[cubic-bezier(0.2,0,0,1)]",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-md-primary",
        "disabled:pointer-events-none disabled:opacity-40",
        VARIANTS[variant],
        SIZES[size],
        fullWidth && "w-full",
        className,
      )}
      {...props}
    >
      {loading ? <Spinner size={18} /> : leadingIcon}
      {children}
    </button>
  );
});
