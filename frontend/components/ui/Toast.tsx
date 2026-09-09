"use client";

import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";
import { CheckCircle2, AlertTriangle, Info, X } from "lucide-react";
import { cn } from "@/lib/cn";

type Kind = "success" | "error" | "info";
interface Toast {
  id: number;
  message: string;
  kind: Kind;
}

interface Api {
  show: (message: string, kind?: Kind) => void;
  success: (message: string) => void;
  error: (message: string) => void;
}

const ToastContext = createContext<Api | null>(null);

const ICON: Record<Kind, typeof Info> = {
  success: CheckCircle2,
  error: AlertTriangle,
  info: Info,
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const remove = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const show = useCallback(
    (message: string, kind: Kind = "info") => {
      const id = Date.now() + Math.random();
      setToasts((t) => [...t, { id, message, kind }]);
      setTimeout(() => remove(id), kind === "error" ? 6000 : 4000);
    },
    [remove],
  );

  const api: Api = {
    show,
    success: (m) => show(m, "success"),
    error: (m) => show(m, "error"),
  };

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        className="pointer-events-none fixed inset-x-0 bottom-0 z-[60] flex flex-col items-center gap-2 p-4 sm:items-end"
        aria-live="polite"
        aria-atomic="false"
      >
        {toasts.map((t) => {
          const Icon = ICON[t.kind];
          return (
            <div
              key={t.id}
              role={t.kind === "error" ? "alert" : "status"}
              className={cn(
                "pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-xl px-4 py-3 elev-3",
                "motion-safe:animate-[fade_150ms_ease-out]",
                t.kind === "error"
                  ? "bg-md-error-container text-md-on-error-container"
                  : "bg-md-surface-container-highest text-md-on-surface",
              )}
            >
              <Icon
                size={18}
                className={cn(
                  "mt-0.5 shrink-0",
                  t.kind === "success" && "text-md-success",
                  t.kind === "info" && "text-md-primary",
                )}
              />
              <span className="t-body-m flex-1">{t.message}</span>
              <button
                type="button"
                aria-label="Dismiss"
                onClick={() => remove(t.id)}
                className="shrink-0 rounded-full p-0.5 opacity-70 hover:opacity-100"
              >
                <X size={16} />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): Api {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}
