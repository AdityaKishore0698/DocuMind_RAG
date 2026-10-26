"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import {
  ChatMessageItem,
  ChatSessionItem,
  DocumentItem,
  getCapabilities,
  getSessionMessages,
  listSessions,
} from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ComparePanel from "@/components/ComparePanel";
import { cn } from "@/lib/cn";

export default function AppShell() {
  const { session } = useAuth();
  const token = session?.access_token ?? null;
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [view, setView] = useState<"chat" | "compare">("chat");
  // Once opened, the comparison stays mounted so its (slow) results survive a trip back to the chat.
  const [compareOpened, setCompareOpened] = useState(false);
  const [agentAvailable, setAgentAvailable] = useState(false);

  useEffect(() => {
    if (!token) return;
    getCapabilities(token)
      .then((c) => setAgentAvailable(c.agent))
      .catch(() => setAgentAvailable(false));
  }, [token]);

  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  // Only ready papers can stay selected (a deleted or failed paper drops out).
  const selectedDocs = useMemo(
    () => docs.filter((d) => d.status === "completed" && d.searchable !== false && selectedIds.includes(d.id)),
    [docs, selectedIds],
  );
  // Desktop-only: the permanent sidebar can be collapsed to an icon rail.
  // Mobile keeps the modal-drawer behaviour untouched.
  const [railCollapsed, setRailCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem("documind-sidebar-collapsed") === "1";
    } catch {
      return false;
    }
  });

  const toggleRail = useCallback(() => {
    setRailCollapsed((v) => {
      const next = !v;
      try {
        window.localStorage.setItem(
          "documind-sidebar-collapsed",
          next ? "1" : "0",
        );
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const refreshSessions = useCallback(async () => {
    if (!token) return;
    try {
      setSessions(await listSessions(token));
    } catch {
      /* handled globally on 401 */
    } finally {
      setSessionsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshSessions();
  }, [refreshSessions]);

  const openSession = useCallback(
    async (id: number) => {
      if (!token) return;
      setView("chat");
      setActiveSessionId(id);
      // Papers are chat-scoped: neither a selection nor a comparison carries over to another chat.
      setSelectedIds([]);
      setCompareOpened(false);
      setDrawerOpen(false);
      setMessagesLoading(true);
      try {
        setMessages(await getSessionMessages(token, id));
      } catch {
        setMessages([]);
      } finally {
        setMessagesLoading(false);
      }
    },
    [token],
  );

  const newChat = useCallback(() => {
    setView("chat");
    setActiveSessionId(null);
    setSelectedIds([]);
    setCompareOpened(false);
    setMessages([]);
    setDrawerOpen(false);
  }, []);

  const handleSessionCreated = useCallback(
    (id: number) => {
      setActiveSessionId(id);
      refreshSessions();
    },
    [refreshSessions],
  );

  const handleSessionRemoved = useCallback(
    (id: number) => {
      if (id === activeSessionId) newChat();
      refreshSessions();
    },
    [activeSessionId, newChat, refreshSessions],
  );

  return (
    <div className="flex h-dvh overflow-hidden bg-md-surface text-md-on-surface">
      <Sidebar
        sessions={sessions}
        sessionsLoading={sessionsLoading}
        activeSessionId={activeSessionId}
        open={drawerOpen}
        collapsed={railCollapsed}
        onToggleCollapsed={toggleRail}
        onClose={() => setDrawerOpen(false)}
        onNewChat={newChat}
        onOpenSession={openSession}
        onSessionsChanged={refreshSessions}
        onSessionRemoved={handleSessionRemoved}
        onSessionCreated={handleSessionCreated}
        selectedIds={selectedIds}
        onToggleSelect={toggleSelect}
        onDocsChanged={setDocs}
        onOpenCompare={() => {
          setView("compare");
          setCompareOpened(true);
          setDrawerOpen(false);
        }}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        {compareOpened && activeSessionId !== null && (
          <div className={cn("flex min-h-0 flex-1 flex-col", view !== "compare" && "hidden")}>
            <ComparePanel
              key={activeSessionId}
              sessionId={activeSessionId}
              selectedDocs={selectedDocs}
              onBack={() => setView("chat")}
              onToggleDrawer={() => setDrawerOpen((v) => !v)}
            />
          </div>
        )}
        <div className={cn("flex min-h-0 flex-1 flex-col", view !== "chat" && "hidden")}>
          <ChatPanel
            key={activeSessionId ?? "new"}
            activeSessionId={activeSessionId}
            messages={messages}
            messagesLoading={messagesLoading}
            setMessages={setMessages}
            onSessionCreated={handleSessionCreated}
            onToggleDrawer={() => setDrawerOpen((v) => !v)}
            selectedDocs={selectedDocs}
            onClearSelection={() => setSelectedIds([])}
            agentAvailable={agentAvailable}
          />
        </div>
      </main>
    </div>
  );
}
