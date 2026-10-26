"use client";

import { useState } from "react";
import { ChevronDown, FileText } from "lucide-react";
import type { SourceItem } from "@/lib/api";
import { cn } from "@/lib/cn";

export function sourceLabel(s: SourceItem): string {
  const parts = [s.title || s.filename || `Document ${s.document_id}`];
  if (s.page_number) parts.push(`p. ${s.page_number}`);
  if (s.section) parts.push(s.section);
  return parts.join(" · ");
}

/** A small [S n] chip; hover/focus shows where the evidence comes from. */
export function CitationChip({
  n,
  source,
  prefix = "S",
}: {
  n: number;
  source?: SourceItem;
  /** Comparison tables number each paper's sources separately: "A1" = paper A's source 1. */
  prefix?: string;
}) {
  return (
    <span
      tabIndex={0}
      title={source ? `${sourceLabel(source)}\n\n${source.snippet}` : "Unknown source"}
      className={cn(
        "inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 align-middle t-label-s tabular-nums",
        source
          ? "bg-md-primary-container text-md-on-primary-container"
          : "bg-md-error-container text-md-on-error-container",
      )}
    >
      {prefix}
      {n}
    </span>
  );
}

/** Collapsible list of the sources an answer was given, cited ones first-class. */
export function SourceList({
  sources,
  cited,
}: {
  sources: SourceItem[];
  cited: number[];
}) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;
  const citedSet = new Set(cited);
  const shown = open ? sources : sources.filter((s) => citedSet.has(s.n));
  return (
    <div className="mt-3 border-t border-md-outline-variant pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-center gap-1 rounded-md t-label-m text-md-on-surface-variant hover:text-md-on-surface"
      >
        <ChevronDown size={14} className={cn("transition-transform", open && "rotate-180")} />
        {cited.length > 0
          ? `Cited ${cited.length} of ${sources.length} sources`
          : `${sources.length} sources retrieved — none cited`}
      </button>
      {shown.length > 0 && (
        <ol className="mt-2 space-y-1.5">
          {shown.map((s) => (
            <li key={s.n} className="flex gap-2 t-body-s">
              <CitationChip n={s.n} source={s} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1 text-md-on-surface">
                  <FileText size={12} className="shrink-0 opacity-60" />
                  <span className="truncate" title={sourceLabel(s)}>
                    {sourceLabel(s)}
                  </span>
                </div>
                {open && (
                  <p className="mt-0.5 line-clamp-3 text-md-on-surface-variant">{s.snippet}</p>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
