"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Columns3,
  FileText,
  LogOut,
  MessageSquarePlus,
  MessageSquare,
  RotateCcw,
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
  assignDocument,
  createSession,
  deleteAccount,
  deleteDocument,
  deleteSession,
  listDocuments,
  listUnassignedDocuments,
  reprocessDocument,
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
  /** A chat was created here (by uploading papers before asking anything). */
  onSessionCreated: (id: number) => void;
  /** Papers ticked for focused chat / comparison. */
  selectedIds: number[];
  onToggleSelect: (id: number) => void;
  onDocsChanged: (docs: DocumentItem[]) => void;
  onOpenCompare: () => void;
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
  onSessionCreated,
  selectedIds,
  onToggleSelect,
  onDocsChanged,
  onOpenCompare,
}: Props) {
  const { session, signOut } = useAuth();
  const token = session?.access_token ?? null;
  const toast = useToast();

  // Papers are chat-scoped: the list shows only the active chat's papers (none for a new chat).
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [legacy, setLegacy] = useState<DocumentItem[]>([]);
  const [pendingDelete, setPendingDelete] = useState<ChatSessionItem | null>(null);
  const [deletingChat, setDeletingChat] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [renamingId, setRenamingId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteAcctOpen, setDeleteAcctOpen] = useState(false);
  const [deletingAcct, setDeletingAcct] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // The chat currently shown; async refreshes for a chat that is no longer active are discarded.
  const activeRef = useRef(activeSessionId);
  useEffect(() => {
    activeRef.current = activeSessionId;
  }, [activeSessionId]);

  /** `sessionId` defaults to the active chat; an upload that just created a chat passes its new id, because this
   *  callback's closure may still hold the previous (null) chat. */
  const refreshDocs = useCallback(async (sessionId: number | null = activeSessionId) => {
    if (!token) return;
    try {
      const [list, unassigned] = await Promise.all([
        sessionId === null ? Promise.resolve([]) : listDocuments(token, sessionId),
        listUnassignedDocuments(token),
      ]);
      setLegacy(unassigned);
      if (sessionId !== activeRef.current) return;
      setDocs(list);
      onDocsChanged(list);
    } catch {
      /* 401 handled globally */
    } finally {
      setDocsLoading(false);
    }
  }, [token, activeSessionId, onDocsChanged]);

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
      // A new chat gets created by its first upload, so the papers belong to it from the start.
      let sessionId = activeSessionId;
      if (sessionId === null) {
        sessionId = (await createSession(token, "New chat")).id;
        onSessionCreated(sessionId);
      }
      const results = await uploadDocuments(token, sessionId, Array.from(files));
      activeRef.current = sessionId; // the parent switches to the new chat; don't wait for the next render
      toast.success(
        results.length === 1
          ? `“${results[0].filename}” uploaded — processing now.`
          : `${results.length} files uploaded — processing now.`,
      );
      await refreshDocs(sessionId);
      onSessionsChanged(); // paper counts shown when deleting a chat
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function reprocess(id: number, name: string) {
    if (!token || activeSessionId === null) return;
    try {
      await reprocessDocument(token, id, activeSessionId);
      toast.success(`Re-processing “${name}”.`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not re-process that document.");
    } finally {
      refreshDocs();
    }
  }

  async function removeDoc(id: number, name: string, sessionId: number | null) {
    if (!token) return;
    setDocs((prev) => prev.filter((d) => d.id !== id));
    setLegacy((prev) => prev.filter((d) => d.id !== id));
    if (selectedIds.includes(id)) onToggleSelect(id);
    try {
      const result = await deleteDocument(token, id, sessionId);
      if (result?.files_not_removed) toast.error(`Removed “${name}”, but its stored file could not be deleted.`);
      else toast.success(`Removed “${name}”.`);
      onSessionsChanged();
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

  async function moveHere(id: number, name: string) {
    if (!token || activeSessionId === null) return;
    try {
      await assignDocument(token, id, activeSessionId);
      toast.success(`Moved “${name}” into this chat.`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not move that paper.");
    } finally {
      refreshDocs();
      onSessionsChanged();
    }
  }

  async function confirmDeleteChat() {
    if (!token || !pendingDelete) return;
    const id = pendingDelete.id;
    setDeletingChat(true);
    try {
      const result = await deleteSession(token, id);
      if (result?.files_not_removed) {
        toast.error(
          `Chat deleted, but ${result.files_not_removed} stored file${result.files_not_removed === 1 ? "" : "s"} could not be removed.`,
        );
      } else {
        toast.success("Chat deleted.");
      }
      onSessionRemoved(id);
    } catch (err) {
      // Nothing was deleted: stay in the chat and say so, rather than navigating away as if it had worked.
      toast.error(err instanceof Error ? `Could not delete the chat: ${err.message}` : "Could not delete the chat.");
      onSessionsChanged();
    } finally {
      setDeletingChat(false);
      setPendingDelete(null);
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
          <span className="t-title-l">ResearchLens</span>
          <div className="flex items-center gap-1">
            <IconButton
              label="Collapse sidebar"
              className="max-md:hidden"
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
              Papers in this chat
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
            {selectedIds.length > 0 && (
              <Button
                variant="outlined"
                fullWidth
                leadingIcon={<Columns3 size={18} />}
                disabled={selectedIds.length < 2 || selectedIds.length > 6}
                onClick={onOpenCompare}
                className="mt-2"
              >
                {selectedIds.length < 2
                  ? "Tick another paper to compare"
                  : selectedIds.length > 6
                    ? "Compare up to 6 papers"
                    : `Compare ${selectedIds.length} papers`}
              </Button>
            )}

            <ul className="mt-3 space-y-0.5">
              {docsLoading &&
                Array.from({ length: 3 }).map((_, i) => (
                  <li key={i} className="px-2 py-2">
                    <Skeleton className="h-4 w-full" />
                  </li>
                ))}
              {!docsLoading && docs.length === 0 && (
                <li className="px-2 py-2 t-body-m text-md-on-surface-variant">
                  {activeSessionId === null
                    ? "Upload papers to start a chat. Papers stay in the chat they are uploaded in."
                    : "No papers in this chat yet. Papers uploaded here are only used in this chat."}
                </li>
              )}
              {docs.map((doc) => {
                const s = STATUS[doc.status] ?? {
                  label: doc.status,
                  className: "bg-md-surface-container-high text-md-on-surface-variant",
                };
                const otherSpace = doc.status === "completed" && doc.searchable === false;
                const ready = doc.status === "completed" && !otherSpace;
                const name = doc.title || doc.filename;
                const byline = [doc.authors, doc.year].filter(Boolean).join(" · ");
                const note = otherSpace
                  ? `Embedded with ${doc.embedding_model ?? "an unknown model"} — re-process to search it`
                  : doc.error;
                return (
                  <li
                    key={doc.id}
                    className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-md-on-surface/[0.05]"
                  >
                    {ready ? (
                      <input
                        type="checkbox"
                        checked={selectedIds.includes(doc.id)}
                        onChange={() => onToggleSelect(doc.id)}
                        aria-label={`Select ${name}`}
                        className="h-4 w-4 shrink-0 accent-[var(--md-primary)]"
                      />
                    ) : (
                      <FileText size={16} className="shrink-0 text-md-on-surface-variant" />
                    )}
                    <span
                      className="min-w-0 flex-1"
                      title={[name, byline, doc.filename, note].filter(Boolean).join("\n")}
                    >
                      <span className="block truncate t-body-m">{name}</span>
                      {(byline || note) && (
                        <span
                          className={cn(
                            "block truncate t-body-s",
                            note ? "text-md-error" : "text-md-on-surface-variant",
                          )}
                        >
                          {note || byline}
                        </span>
                      )}
                    </span>
                    {otherSpace || doc.status === "failed" ? (
                      <button
                        type="button"
                        onClick={() => reprocess(doc.id, name)}
                        className="inline-flex shrink-0 items-center gap-1 rounded-full bg-md-error-container px-2 py-0.5 t-label-s text-md-on-error-container hover:opacity-90"
                      >
                        <RotateCcw size={12} /> Re-process
                      </button>
                    ) : (
                      <span className={cn("shrink-0 rounded-full px-2 py-0.5 t-label-s", s.className)}>
                        {s.label}
                      </span>
                    )}
                    <IconButton
                      label={`Remove ${doc.filename}`}
                      onClick={() => removeDoc(doc.id, doc.filename, activeSessionId)}
                      className="h-8 w-8 opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
                    >
                      <Trash2 size={15} />
                    </IconButton>
                  </li>
                );
              })}
            </ul>
          </section>

          {/* Papers uploaded before papers belonged to chats: searched nowhere until moved into one */}
          {legacy.length > 0 && (
            <section aria-labelledby="legacy-heading">
              <h2 id="legacy-heading" className="px-2 t-label-m uppercase text-md-on-surface-variant">
                Earlier papers (not in a chat)
              </h2>
              <p className="px-2 pt-1 t-body-s text-md-on-surface-variant">
                Uploaded before papers belonged to chats. They are not searched until you move one into a chat.
              </p>
              <ul className="mt-2 space-y-0.5">
                {legacy.map((doc) => {
                  const name = doc.title || doc.filename;
                  return (
                    <li
                      key={doc.id}
                      className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-md-on-surface/[0.05]"
                    >
                      <FileText size={16} className="shrink-0 text-md-on-surface-variant" />
                      <span className="min-w-0 flex-1 truncate t-body-m" title={name}>
                        {name}
                      </span>
                      <button
                        type="button"
                        disabled={activeSessionId === null}
                        title={activeSessionId === null ? "Open or start a chat first" : "Move into this chat"}
                        onClick={() => moveHere(doc.id, name)}
                        className="inline-flex shrink-0 items-center gap-1 rounded-full bg-md-secondary-container px-2 py-0.5 t-label-s text-md-on-secondary-container hover:opacity-90 disabled:opacity-40"
                      >
                        <ArrowDownToLine size={12} /> Move here
                      </button>
                      <IconButton
                        label={`Remove ${doc.filename}`}
                        onClick={() => removeDoc(doc.id, doc.filename, null)}
                        className="h-8 w-8 opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
                      >
                        <Trash2 size={15} />
                      </IconButton>
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

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
                        onClick={() => setPendingDelete(sn)}
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
        open={pendingDelete !== null}
        onClose={() => setPendingDelete(null)}
        title="Delete this chat?"
        confirmLabel="Delete chat"
        destructive
        loading={deletingChat}
        onConfirm={confirmDeleteChat}
      >
        {pendingDelete?.paper_count
          ? `This permanently deletes the chat, its conversation history, ${
              pendingDelete.paper_count === 1
                ? "the paper uploaded in it and its stored file"
                : `the ${pendingDelete.paper_count} papers uploaded in it and their stored files`
            }. Papers in other chats are not affected. This cannot be undone.`
          : "This permanently deletes the chat and its conversation history. It has no papers. This cannot be undone."}
      </Dialog>

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
