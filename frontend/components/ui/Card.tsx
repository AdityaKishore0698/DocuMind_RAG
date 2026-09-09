import { type HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

type Variant = "elevated" | "filled" | "outlined";

const VARIANTS: Record<Variant, string> = {
  elevated: "bg-md-surface-container-low elev-1",
  filled: "bg-md-surface-container-highest",
  outlined: "bg-md-surface border border-md-outline-variant",
};

export function Card({
  variant = "elevated",
  className,
  ...props
}: HTMLAttributes<HTMLDivElement> & { variant?: Variant }) {
  return (
    <div
      className={cn("rounded-xl text-md-on-surface", VARIANTS[variant], className)}
      {...props}
    />
  );
}
