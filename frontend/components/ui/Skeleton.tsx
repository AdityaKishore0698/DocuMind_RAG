import { cn } from "@/lib/cn";

export function Skeleton({
  className,
  rounded = "rounded-md",
}: {
  className?: string;
  rounded?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "relative block overflow-hidden bg-md-surface-container-high",
        rounded,
        className,
      )}
    >
      <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-md-on-surface/[0.06] to-transparent motion-safe:animate-[shimmer_1.6s_infinite]" />
    </span>
  );
}
