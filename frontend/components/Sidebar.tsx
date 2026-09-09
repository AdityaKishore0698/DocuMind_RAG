"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileText,
  LogOut,
  MessageSquarePlus,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Trash2,
  Upload,
  UserX,
  X,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import {
  ChatSessionItem,
  DocumentItem,
  deleteAccount,
  deleteDocument,
  deleteSession,
  listDocuments,
  renameSession,
  uploadDocuments,
} from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { IconButton } from "@/components/ui/IconButton";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { cn } from "@/lib/cn";

interface Props {
  sessions: ChatSessionItem[];
  sessionsLoading: boolean;
  activeSessionId: number | null;
  open: boolean;
  /** Desktop (>= md) only: render as a compact icon rail. */
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onClose: () => void;
  onNewChat: () => void;
  onOpenSession: (id: number) => void;
  onSessionsChanged: () => void;
  onSessionRemoved: (id: number) => void;
}

const STATUS: Record<string, { label: string; className: string }> = {
  completed: { label: "Ready", className: "bg-md-primary-container text-md-on-primary-container" },
  processing: { label: "Processing", className: "bg-md-secondary-container text-md-on-secondary-container" },
  uploaded: { label: "Queued", className: "bg-md-secondary-container text-md-on-secondary-container" },
  failed: { label: "Failed", className: "bg-md-error-container text-md-on-error-container" },
};

export default function Sidebar({
  sessions,
  sessionsLoading,
  activeSessionId,
  open,
  collapsed,
  onToggleCollapsed,
  onClose,
  onNewChat,
  onOpenSession,
  onSessionsChanged,
  onSessionRemoved,
}: Props) {
  const { session, signOut } = useAuth();
  const token = session?.access_token ?? null;
  const toast = useToast();

  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [renamingId, setRenamingId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteAcctOpen, setDeleteAcctOpen] = useState(false);
  const [deletingAcct, setDeletingAcct] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshDocs = useCallback(async () => {
    if (!token) return;
    try {
      setDocs(await listDocuments(token));
    } catch {
      /* 401 handled globally */
    } finally {
      setDocsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshDocs();
  }, [refreshDocs]);

  useEffect(() => {
    const pending = docs.some((d) => d.status !== "completed" && d.status !== "failed");
    if (!pending) return;
    const t = setInterval(refreshDocs, 3000);
    return () => clearInterval(t);
  }, [docs, refreshDocs]);

  async function handleFiles(files: FileList | null) {
    if (!token || !files || files.length === 0) return;
    setUploading(true);
    try {
      const results = await uploadDocuments(token, Array.from(files));
      toast.success(
        results.length === 1
          ? `“${results[0].filename}” uploaded — processing now.`
          : `${results.length} files uploaded — processing now.`,
      );
      await refreshDocs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function removeDoc(id: number, name: string) {
    if (!token) return;
    setDocs((prev) => prev.filter((d) => d.id !== id));
    try {
      await deleteDocument(token, id);
      toast.success(`Removed “${name}”.`);
    } catch {
      toast.error("Could not remove that document.");
    } finally {
      refreshDocs();
    }
  }

  async function submitRename(id: number) {
    if (!token || !renameValue.trim()) {
      setRenamingId(null);
      return;
    }
    try {
      await renameSession(token, id, renameValue.trim());
      onSessionsChanged();
    } finally {
      setRenamingId(null);
    }
  }

  async function removeSession(id: number) {
    if (!token) return;
    try {
      await deleteSession(token, id);
    } finally {
      onSessionRemoved(id);
    }
  }

  async function confirmDeleteAccount() {
    if (!token) return;
    setDeletingAcct(true);
    try {
      await deleteAccount(token);
    } finally {
      setDeletingAcct(false);
      setDeleteAcctOpen(false);
      signOut();
    }
  }

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-20 md:hidden"
          style={{ background: "var(--md-scrim)" }}
          onClick={onClose}
          aria-hidden
        />
      )}

      <nav
        aria-label="Documents and conversations"
        className={cn(
          "fixed inset-y-0 left-0 z-30 flex w-[22rem] max-w-[85vw] flex-col bg-md-surface-container-low",
          "transition-[transform,width] duration-[250ms] ease-[cubic-bezier(0.2,0,0,1)]",
          "md:static md:z-auto md:max-w-none md:translate-x-0 md:border-r md:border-md-outline-variant",
          open ? "translate-x-0 elev-2" : "-translate-x-full",
          collapsed ? "md:w-[4.5rem]" : "md:w-80",
        )}
      >
        {/* Full sidebar — always shown on mobile; on desktop only when expanded */}
        <div className={cn("flex min-h-0 flex-1 flex-col", collapsed && "md:hidden")}>
        <div className="flex items-center justify-between px-4 pb-2 pt-4">
          <span className="t-title-l">DocuMind</span>
          <div className="flex items-center gap-1">
            <IconButton
              label="Collapse sidebar"
              className="hidden md:inline-grid"
              onClick={onToggleCollapsed}
            >
              <PanelLeftClose size={20} />
            </IconButton>
            <IconButton label="Close menu" className="md:hidden" onClick={onClose}>
              <X size={20} />
            </IconButton>
          </div>
        </div>

        <div className="flex-1 space-y-7 overflow-y-auto px-3 py-3 scrollbar-thin">
          {/* Knowledge base */}
          <section aria-labelledby="kb-heading">
            <h2 id="kb-heading" className="px-2 t-label-m uppercase text-md-on-surface-variant">
              Knowledge base
            </h2>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <Button
              variant="tonal"
              fullWidth
              loading={uploading}
              leadingIcon={<Upload size={18} />}
              onClick={() => fileInputRef.current?.click()}
              className="mt-2"
            >
              {uploading ? "Uploading…" : "Upload PDF or TXT"}
            </Button>

            <ul className="mt-3 space-y-0.5">
              {docsLoading &&
                Array.from({ length: 3 }).map((_, i) => (
                  <li key={i} className="px-2 py-2">
                    <Skeleton className="h-4 w-full" />
                  </li>
                ))}
              {!docsLoading && docs.length === 0 && (
                <li className="px-2 py-2 t-body-m text-md-on-surface-variant">
                  No documents yet.
                </li>
              )}
              {docs.map((doc) => {
                const s = STATUS[doc.status] ?? {
                  label: doc.status,
                  className: "bg-md-surface-container-high text-md-on-surface-variant",
                };
                return (
                  <li
                    key={doc.id}
                    className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-md-on-surface/[0.05]"
                  >
                    <FileText size={16} className="shrink-0 text-md-on-surface-variant" />
                    <span className="min-w-0 flex-1 truncate t-body-m" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <span className={cn("shrink-0 rounded-full px-2 py-0.5 t-label-s", s.className)}>
                      {s.label}
                    </span>
                    <IconButton
                      label={`Remove ${doc.filename}`}
                      onClick={() => removeDoc(doc.id, doc.filename)}
                      className="h-8 w-8 opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
                    >
                      <Trash2 size={15} />
                    </IconButton>
                  </li>
                );
              })}
            </ul>
          </section>

          {/* Chats */}
          <section aria-labelledby="chats-heading">
            <div className="flex items-center justify-between px-2">
              <h2 id="chats-heading" className="t-label-m uppercase text-md-on-surface-variant">
                Chats
              </h2>
              <button
                type="button"
                onClick={onNewChat}
                className="inline-flex items-center gap-1 rounded-full px-2 py-1 t-label-l text-md-primary hover:bg-md-primary/[0.08]"
              >
                <MessageSquarePlus size={16} /> New
              </button>
            </div>

            <ul className="mt-2 space-y-0.5">
              {sessionsLoading &&
                Array.from({ length: 3 }).map((_, i) => (
                  <li key={i} className="px-2 py-2">
                    <Skeleton className="h-4 w-3/4" />
                  </li>
                ))}
              {!sessionsLoading && sessions.length === 0 && (
                <li className="px-2 py-2 t-body-m text-md-on-surface-variant">
                  No conversations yet.
                </li>
              )}
              {sessions.map((sn) => (
                <li key={sn.id} className="group">
                  {renamingId === sn.id ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onBlur={() => submitRename(sn.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") submitRename(sn.id);
                        if (e.key === "Escape") setRenamingId(null);
                      }}
                      className="w-full rounded-lg border border-md-primary bg-md-surface-container-highest px-2 py-2 t-body-m outline-none"
                    />
                  ) : (
                    <div
                      className={cn(
                        "flex items-center gap-1 rounded-full pl-3 pr-1 py-1",
                        activeSessionId === sn.id
                          ? "bg-md-secondary-container text-md-on-secondary-container"
                          : "hover:bg-md-on-surface/[0.05]",
                      )}
                    >
                      <MessageSquare size={16} className="shrink-0 opacity-70" />
                      <button
                        type="button"
                        onClick={() => onOpenSession(sn.id)}
                        className="min-w-0 flex-1 truncate py-1 text-left t-body-m"
                        title={sn.title}
                      >
                        {sn.title}
                      </button>
                      <IconButton
                        label="Rename chat"
                        onClick={() => {
                          setRenamingId(sn.id);
                          setRenameValue(sn.title);
                        }}
                        className="h-8 w-8 opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
                      >
                        <Pencil size={14} />
                      </IconButton>
                      <IconButton
                        label="Delete chat"
                        onClick={() => removeSession(sn.id)}
                        className="h-8 w-8 opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
                      >
                        <Trash2 size={14} />
                      </IconButton>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </section>
        </div>

        {/* Account */}
        <div className="border-t border-md-outline-variant p-2">
          <button
            type="button"
            onClick={signOut}
            className="flex w-full items-center gap-3 rounded-full px-3 py-2.5 t-label-l text-md-on-surface-variant hover:bg-md-on-surface/[0.06] hover:text-md-on-surface"
          >
            <LogOut size={18} /> Sign out
          </button>
          <button
            type="button"
            onClick={() => setDeleteAcctOpen(true)}
            className="flex w-full items-center gap-3 rounded-full px-3 py-2.5 t-label-l text-md-error hover:bg-md-error/[0.1]"
          >
            <UserX size={18} /> Delete account
          </button>
        </div>
        </div>

        {/* Collapsed icon rail — desktop only */}
        {collapsed && (
          <div className="hidden min-h-0 flex-1 flex-col md:flex">
            <div className="flex flex-col items-center gap-1 px-2 pt-4">
              <IconButton label="Expand sidebar" onClick={onToggleCollapsed}>
                <PanelLeftOpen size={20} />
              </IconButton>
              <IconButton label="New chat" onClick={onNewChat}>
                <MessageSquarePlus size={20} />
              </IconButton>
              <IconButton
                label="Upload PDF or TXT"
                loading={uploading}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload size={20} />
              </IconButton>
            </div>

            <div className="mx-auto my-2 h-px w-8 bg-md-outline-variant" />

            <ul
              aria-label="Conversations"
              className="flex-1 space-y-1 overflow-y-auto px-2 py-1 scrollbar-thin"
            >
              {sessions.map((sn) => (
                <li key={sn.id}>
                  <IconButton
                    label={sn.title}
                    onClick={() => onOpenSession(sn.id)}
                    className={cn(
                      activeSessionId === sn.id &&
                        "bg-md-secondary-container text-md-on-secondary-container",
                    )}
                  >
                    <MessageSquare size={18} />
                  </IconButton>
                </li>
              ))}
            </ul>

            <div className="flex flex-col items-center border-t border-md-outline-variant p-2">
              <IconButton label="Sign out" onClick={signOut}>
                <LogOut size={18} />
              </IconButton>
            </div>
          </div>
        )}
      </nav>

      <Dialog
        open={deleteAcctOpen}
        onClose={() => setDeleteAcctOpen(false)}
        title="Delete your account?"
        confirmLabel="Delete everything"
        destructive
        loading={deletingAcct}
        onConfirm={confirmDeleteAccount}
      >
        This permanently removes your account, every uploaded document, and all chat
        history. It cannot be undone.
      </Dialog>
    </>
  );
}
