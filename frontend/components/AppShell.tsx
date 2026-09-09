"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import {
  ChatMessageItem,
  ChatSessionItem,
  getSessionMessages,
  listSessions,
} from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";

export default function AppShell() {
  const { session } = useAuth();
  const token = session?.access_token ?? null;
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
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
      setActiveSessionId(id);
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
    setActiveSessionId(null);
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
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <ChatPanel
          key={activeSessionId ?? "new"}
          activeSessionId={activeSessionId}
          messages={messages}
          messagesLoading={messagesLoading}
          setMessages={setMessages}
          onSessionCreated={handleSessionCreated}
          onToggleDrawer={() => setDrawerOpen((v) => !v)}
        />
      </main>
    </div>
  );
}
