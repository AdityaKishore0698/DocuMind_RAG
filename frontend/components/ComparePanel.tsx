"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, ArrowLeft, Lightbulb, Menu, Table2 } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import {
  ApiError,
  CompareAspect,
  CompareResult,
  DocumentItem,
  GapsResult,
  comparePapers,
  findGaps,
} from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { IconButton } from "@/components/ui/IconButton";
import { Skeleton } from "@/components/ui/Skeleton";
import { ThemeToggle } from "@/components/ThemeToggle";
import { CitationChip } from "@/components/Sources";
import { cn } from "@/lib/cn";

const ASPECTS: { id: CompareAspect; label: string }[] = [
  { id: "profile", label: "Overview" },
  { id: "methods", label: "Methods" },
  { id: "results", label: "Results" },
  { id: "limitations", label: "Limitations" },
];

const NOT_REPORTED = "not reported in the paper";

function fieldLabel(key: string): string {
  const s = key.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function errorText(err: unknown): string {
  return err instanceof ApiError ? err.message : "Could not reach the API.";
}

interface Props {
  /** The chat whose papers are compared (papers are chat-scoped). */
  sessionId: number;
  selectedDocs: DocumentItem[];
  onBack: () => void;
  onToggleDrawer: () => void;
}

export default function ComparePanel({ sessionId, selectedDocs, onBack, onToggleDrawer }: Props) {
  const { session } = useAuth();
  const token = session?.access_token ?? null;
  const [aspect, setAspect] = useState<CompareAspect>("profile");
  const [table, setTable] = useState<CompareResult | null>(null);
  const [gaps, setGaps] = useState<GapsResult | null>(null);
  const [busy, setBusy] = useState<"table" | "gaps" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const ids = selectedDocs.map((d) => d.id);
  const ready = ids.length >= 2 && ids.length <= 6;
  const key = [...ids].sort((a, b) => a - b).join(",");
  const keyOf = (papers: { document_id: number }[]) =>
    papers.map((p) => p.document_id).sort((a, b) => a - b).join(",");
  const tableStale = table !== null && keyOf(table.papers) !== key;
  const gapsStale = gaps !== null && keyOf(gaps.papers) !== key;

  useEffect(() => () => abortRef.current?.abort(), []);

  async function run(kind: "table" | "gaps", nextAspect = aspect) {
    if (!token || !ready) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(kind);
    setError(null);
    try {
      if (kind === "table") setTable(await comparePapers(token, sessionId, ids, nextAspect, controller.signal));
      else setGaps(await findGaps(token, sessionId, ids, controller.signal));
    } catch (err) {
      if (!controller.signal.aborted) setError(errorText(err));
    } finally {
      if (abortRef.current === controller) {
        setBusy(null);
        abortRef.current = null;
      }
    }
  }

  return (
    <>
      <header className="flex items-center gap-2 border-b border-md-outline-variant bg-md-surface px-3 py-2 md:px-4">
        <IconButton label="Open menu" className="md:hidden" onClick={onToggleDrawer}>
          <Menu size={20} />
        </IconButton>
        <IconButton label="Back to chat" onClick={onBack}>
          <ArrowLeft size={20} />
        </IconButton>
        <h1 className="min-w-0 flex-1 truncate t-title-m">Compare papers</h1>
        <ThemeToggle className="-mr-1 shrink-0" />
      </header>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6">
          {!ready ? (
            <p className="rounded-xl bg-md-surface-container p-4 t-body-m text-md-on-surface-variant">
              Tick between two and six ready papers in the sidebar to compare them.
            </p>
          ) : (
            <p className="t-body-m text-md-on-surface-variant">
              Comparing{" "}
              <span className="text-md-on-surface">
                {selectedDocs.map((d) => d.title || d.filename).join(" · ")}
              </span>
            </p>
          )}

          <section aria-labelledby="cmp-heading" className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <h2 id="cmp-heading" className="mr-auto t-title-m">
                Side-by-side table
              </h2>
              <div role="radiogroup" aria-label="Aspect" className="flex rounded-full border border-md-outline">
                {ASPECTS.map((a) => (
                  <button
                    key={a.id}
                    type="button"
                    role="radio"
                    aria-checked={aspect === a.id}
                    onClick={() => setAspect(a.id)}
                    className={cn(
                      "px-3 py-1.5 t-label-l first:rounded-l-full last:rounded-r-full",
                      aspect === a.id
                        ? "bg-md-secondary-container text-md-on-secondary-container"
                        : "text-md-on-surface-variant hover:bg-md-on-surface/[0.06]",
                    )}
                  >
                    {a.label}
                  </button>
                ))}
              </div>
              <Button
                variant="filled"
                loading={busy === "table"}
                disabled={!ready || busy !== null}
                leadingIcon={<Table2 size={18} />}
                onClick={() => run("table")}
              >
                Compare
              </Button>
            </div>

            {busy === "table" && <Skeleton className="h-48 w-full" rounded="rounded-xl" />}
            {tableStale && busy !== "table" && <StaleNote />}
            {table && busy !== "table" && <CompareTable result={table} />}
          </section>

          <section aria-labelledby="gaps-heading" className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <h2 id="gaps-heading" className="mr-auto t-title-m">
                Research gaps
              </h2>
              <Button
                variant="tonal"
                loading={busy === "gaps"}
                disabled={!ready || busy !== null}
                leadingIcon={<Lightbulb size={18} />}
                onClick={() => run("gaps")}
              >
                Suggest gaps
              </Button>
            </div>
            {busy === "gaps" && <Skeleton className="h-32 w-full" rounded="rounded-xl" />}
            {gapsStale && busy !== "gaps" && <StaleNote />}
            {gaps && busy !== "gaps" && <GapList result={gaps} />}
          </section>

          {error && (
            <p role="alert" className="rounded-xl bg-md-error-container p-3 t-body-m text-md-on-error-container">
              {error}
            </p>
          )}
        </div>
      </div>
    </>
  );
}

function StaleNote() {
  return (
    <p className="flex items-start gap-2 t-body-s text-md-warning">
      <AlertTriangle size={14} className="mt-0.5 shrink-0" /> The paper selection has changed since this was
      produced. Run it again to include the current selection.
    </p>
  );
}

function CompareTable({ result }: { result: CompareResult }) {
  const fields = Object.keys(result.fields);
  // Each paper's sources are numbered separately, so chips carry the column letter (A1, B1, …).
  const letterOf = (i: number) => String.fromCharCode(65 + i);
  return (
    <div className="flex flex-col gap-2">
      {result.warnings.map((w) => (
        <p key={w} className="flex items-start gap-2 t-body-s text-md-warning">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {w}
        </p>
      ))}
      <div className="overflow-x-auto rounded-xl border border-md-outline-variant">
        <table className="w-full min-w-[40rem] border-collapse t-body-m">
          <thead>
            <tr className="bg-md-surface-container">
              <th scope="col" className="w-40 p-3 text-left t-label-m uppercase text-md-on-surface-variant">
                Field
              </th>
              {result.papers.map((p, i) => (
                <th key={p.document_id} scope="col" className="p-3 text-left align-top">
                  <div className="t-title-s">
                    <span className="text-md-primary">{letterOf(i)}</span> · {p.title}
                  </div>
                  <div className="t-body-s text-md-on-surface-variant">
                    {[p.authors, p.year].filter(Boolean).join(" · ") || p.filename}
                  </div>
                  {p.error && <div className="mt-1 t-body-s text-md-error">{p.error}</div>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => (
              <tr key={f} className="border-t border-md-outline-variant">
                <th scope="row" className="p-3 text-left align-top t-label-l" title={result.fields[f]}>
                  {fieldLabel(f)}
                </th>
                {result.papers.map((p, i) => {
                  const cell = p.fields[f];
                  const missing = !cell || cell.value === NOT_REPORTED;
                  return (
                    <td key={p.document_id} className="p-3 align-top">
                      {missing ? (
                        <span className="italic text-md-on-surface-variant">Not reported in the paper</span>
                      ) : (
                        <>
                          {cell.value}{" "}
                          <span className="inline-flex flex-wrap gap-1">
                            {cell.citations.map((n) => (
                              <CitationChip
                                key={n}
                                n={n}
                                prefix={letterOf(i)}
                                source={p.sources.find((s) => s.n === n)}
                              />
                            ))}
                          </span>
                        </>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="t-body-s text-md-on-surface-variant">
        Hover a citation to see the page and passage it comes from. Values without supporting
        evidence are shown as not reported.
      </p>
    </div>
  );
}

function GapList({ result }: { result: GapsResult }) {
  const cross = result.gaps.filter((g) => g.label === "cross_paper_hypothesis");
  const single = result.gaps.filter((g) => g.label === "single_paper_observation");
  const letter = new Map(result.papers.map((p) => [p.document_id, p]));
  return (
    <div className="flex flex-col gap-4">
      <p className="flex items-start gap-2 rounded-xl bg-md-surface-container p-3 t-body-s text-md-on-surface-variant">
        <AlertTriangle size={14} className="mt-0.5 shrink-0 text-md-warning" /> {result.disclaimer}
      </p>
      {result.warnings.map((w) => (
        <p key={w} className="t-body-s text-md-on-surface-variant">
          {w.charAt(0).toUpperCase() + w.slice(1)}.
        </p>
      ))}
      {result.gaps.length === 0 && (
        <p className="t-body-m text-md-on-surface-variant">
          No well-supported gaps were found in the selected papers.
        </p>
      )}
      <GapGroup
        title="Cross-paper hypotheses"
        hint="Each draws on evidence from two or more of the selected papers."
        badge="Hypothesis"
        badgeClass="bg-md-primary-container text-md-on-primary-container"
        gaps={cross}
        result={result}
        letter={letter}
      />
      <GapGroup
        title="Single-paper observations"
        hint="Supported by one paper only — what that paper says about its own limits, not a cross-paper finding."
        badge="One paper"
        badgeClass="bg-md-surface-container-high text-md-on-surface-variant"
        gaps={single}
        result={result}
        letter={letter}
      />
    </div>
  );
}

function GapGroup({
  title,
  hint,
  badge,
  badgeClass,
  gaps,
  result,
  letter,
}: {
  title: string;
  hint: string;
  badge: string;
  badgeClass: string;
  gaps: GapsResult["gaps"];
  result: GapsResult;
  letter: Map<number, GapsResult["papers"][number]>;
}) {
  if (gaps.length === 0) return null;
  return (
    <section className="flex flex-col gap-2">
      <div>
        <h3 className="t-title-s">{title}</h3>
        <p className="t-body-s text-md-on-surface-variant">{hint}</p>
      </div>
      <ul className="flex flex-col gap-2">
        {gaps.map((g, i) => (
          <li key={i} className="rounded-xl border border-md-outline-variant p-3">
            <div className="flex items-start gap-2">
              <span className={cn("shrink-0 rounded-full px-2 py-0.5 t-label-s", badgeClass)}>{badge}</span>
              <p className="t-body-l">{g.gap}</p>
            </div>
            {g.rationale && <p className="mt-1 t-body-m text-md-on-surface-variant">{g.rationale}</p>}
            <div className="mt-2 flex flex-wrap items-center gap-1">
              {g.citations.map((n) => (
                <CitationChip key={n} n={n} source={result.sources.find((s) => s.n === n)} />
              ))}
              <span className="ml-1 t-body-s text-md-on-surface-variant">
                {g.papers
                  .map((id) => {
                    const paper = letter.get(id);
                    return paper ? `Paper ${paper.letter}: ${paper.title}` : `Document ${id}`;
                  })
                  .join(" · ")}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
