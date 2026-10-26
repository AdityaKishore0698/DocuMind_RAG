"use client";

import {
  Dispatch,
  SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BrainCircuit, Menu, Send, Square, X } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import {
  ApiError,
  ChatMessageItem,
  DocumentItem,
  MessageSources,
  citedNumbers,
  streamChat,
} from "@/lib/api";
import { SourceList } from "@/components/Sources";
import { IconButton } from "@/components/ui/IconButton";
import { Skeleton } from "@/components/ui/Skeleton";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/cn";

interface Props {
  activeSessionId: number | null;
  messages: ChatMessageItem[];
  messagesLoading: boolean;
  setMessages: Dispatch<SetStateAction<ChatMessageItem[]>>;
  onSessionCreated: (id: number) => void;
  onToggleDrawer: () => void;
  /** Papers the user ticked in the sidebar; retrieval is restricted to them. */
  selectedDocs: DocumentItem[];
  onClearSelection: () => void;
  agentAvailable: boolean;
}

export default function ChatPanel({
  activeSessionId,
  messages,
  messagesLoading,
  setMessages,
  onSessionCreated,
  onToggleDrawer,
  selectedDocs,
  onClearSelection,
  agentAvailable,
}: Props) {
  const { session } = useAuth();
  const token = session?.access_token ?? null;
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [deep, setDeep] = useState(false);
  const deepMode = deep && agentAvailable;
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  function autosize() {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }

  async function send() {
    const query = input.trim();
    const firstQuestion = messages.length === 0;
    if (!query || streaming || !token) return;

    setInput("");
    requestAnimationFrame(autosize);
    setStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "" },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;
    let acc = "";
    let meta: MessageSources | null = null;

    const writeAssistant = (text: string) =>
      setMessages((prev) => {
        const next = [...prev];
        const sources = meta
          ? { ...meta, cited: citedNumbers(text, meta.sources.length) }
          : null;
        next[next.length - 1] = { role: "assistant", content: text, sources };
        return next;
      });

    try {
      const outcome = await streamChat(
        token,
        {
          query,
          session_id: activeSessionId,
          document_ids: selectedDocs.map((d) => d.id),
          mode: deepMode ? "agent" : "rag",
        },
        {
          signal: controller.signal,
          onSources: (sources, mode) => {
            meta = { sources, cited: [], invalid: [], mode };
          },
          onToken: (chunk) => {
            acc += chunk;
            writeAssistant(acc);
          },
        },
      );
      if (!acc) writeAssistant(outcome.text || "_(no response)_");
      // A new chat, or one created by an upload, is named after its first question: refresh the chat list.
      if (outcome.sessionId && (!activeSessionId || firstQuestion)) onSessionCreated(outcome.sessionId);
    } catch (err) {
      if (controller.signal.aborted) {
        writeAssistant(acc ? `${acc}\n\n_(stopped)_` : "_(stopped)_");
      } else {
        const msg =
          err instanceof ApiError ? `⚠️ ${err.message}` : "⚠️ Could not reach the API.";
        writeAssistant(acc ? `${acc}\n\n${msg}` : msg);
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  return (
    <>
      <header className="flex items-center gap-2 border-b border-md-outline-variant bg-md-surface px-3 py-2 md:px-4">
        <IconButton label="Open menu" className="md:hidden" onClick={onToggleDrawer}>
          <Menu size={20} />
        </IconButton>
        <h1 className="min-w-0 flex-1 truncate t-title-m">
          {activeSessionId ? "Conversation" : "New chat"}
        </h1>
        {selectedDocs.length > 0 && (
          <span
            className="inline-flex max-w-[9rem] items-center gap-1 rounded-full bg-md-secondary-container py-1 pl-3 pr-1 t-label-m text-md-on-secondary-container sm:max-w-[16rem]"
            title={selectedDocs.map((d) => d.title || d.filename).join("\n")}
          >
            <span className="truncate">
              {selectedDocs.length === 1
                ? selectedDocs[0].title || selectedDocs[0].filename
                : `${selectedDocs.length} papers selected`}
            </span>
            <button
              type="button"
              aria-label="Search all papers"
              onClick={onClearSelection}
              className="grid h-5 w-5 shrink-0 place-items-center rounded-full hover:bg-md-on-surface/[0.08]"
            >
              <X size={12} />
            </button>
          </span>
        )}
        <ThemeToggle className="-mr-1 shrink-0" />
      </header>

      <div ref={scrollRef} className="flex-1 overflow-x-hidden overflow-y-auto scrollbar-thin">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6">
          {messagesLoading &&
            Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className={cn("flex gap-3", i % 2 === 1 && "flex-row-reverse")}>
                <Skeleton className="h-8 w-8" rounded="rounded-full" />
                <Skeleton className="h-16 w-2/3" rounded="rounded-2xl" />
              </div>
            ))}

          {!messagesLoading && messages.length === 0 && (
            <div className="mt-16 text-center sm:mt-24">
              <h2 className="t-headline-s">Ask anything about your papers</h2>
              <p className="mt-2 t-body-m text-md-on-surface-variant">
                Upload PDFs from the menu — they belong to this chat only. Tick papers to focus on them, or compare two or more.
              </p>
            </div>
          )}

          {messages.map((m, i) => (
            <Message
              key={i}
              role={m.role}
              content={m.content}
              sources={m.sources ?? null}
              pending={
                streaming &&
                i === messages.length - 1 &&
                m.role === "assistant" &&
                m.content === ""
              }
            />
          ))}
        </div>
      </div>

      <div className="border-t border-md-outline-variant bg-md-surface px-4 py-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="mx-auto flex w-full max-w-3xl items-end gap-2"
        >
          <div className="flex min-w-0 flex-1 items-end rounded-3xl bg-md-surface-container-high px-4 py-1.5 focus-within:outline focus-within:outline-2 focus-within:outline-md-primary">
            <textarea
              ref={taRef}
              value={input}
              rows={1}
              onChange={(e) => {
                setInput(e.target.value);
                autosize();
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask about your papers…"
              aria-label="Message"
              className="max-h-40 min-h-[36px] flex-1 resize-none bg-transparent py-2 t-body-l text-md-on-surface outline-none placeholder:text-md-on-surface-variant"
            />
          </div>
          {agentAvailable && (
            <IconButton
              label={deepMode ? "Deep analysis on" : "Deep analysis off"}
              variant={deepMode ? "filled" : "standard"}
              type="button"
              aria-pressed={deepMode}
              onClick={() => setDeep((v) => !v)}
              disabled={streaming}
            >
              <BrainCircuit size={18} />
            </IconButton>
          )}
          {streaming ? (
            <IconButton
              label="Stop generating"
              variant="tonal"
              type="button"
              onClick={() => abortRef.current?.abort()}
            >
              <Square size={18} fill="currentColor" />
            </IconButton>
          ) : (
            <IconButton
              label="Send message"
              variant="filled"
              type="submit"
              disabled={!input.trim()}
            >
              <Send size={18} />
            </IconButton>
          )}
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center t-label-s text-md-on-surface-variant">
          {deepMode
            ? "Deep analysis plans searches across your papers before answering — slower, more thorough."
            : "Answers cite your uploaded papers as [S1], [S2] … and may be incomplete — check the sources."}
        </p>
      </div>
    </>
  );
}

function Message({
  role,
  content,
  sources,
  pending,
}: {
  role: "user" | "assistant";
  content: string;
  sources: MessageSources | null;
  pending: boolean;
}) {
  const isUser = role === "user";
  return (
    <div className={cn("flex min-w-0 gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "grid h-8 w-8 shrink-0 place-items-center rounded-full t-label-m",
          isUser
            ? "bg-md-primary text-md-on-primary"
            : "bg-md-secondary-container text-md-on-secondary-container",
        )}
        aria-hidden
      >
        {isUser ? "You" : "AI"}
      </div>
      <div
        className={cn(
          "min-w-0 max-w-[85%] overflow-hidden break-words rounded-2xl px-4 py-2.5 t-body-l leading-relaxed",
          isUser
            ? "whitespace-pre-wrap bg-md-primary text-md-on-primary"
            : "bg-md-surface-container text-md-on-surface",
        )}
      >
        {isUser ? (
          content
        ) : (
          <div className="markdown" aria-live={pending ? "polite" : undefined}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            {pending && (
              <span className="ml-0.5 inline-block motion-safe:animate-blink" aria-hidden>
                ▋
              </span>
            )}
            {sources && !pending && (
              <>
                {sources.mode === "agent" && (
                  <span className="mt-2 inline-flex items-center gap-1 t-label-s text-md-on-surface-variant">
                    <BrainCircuit size={12} /> Deep analysis
                  </span>
                )}
                <SourceList sources={sources.sources} cited={sources.cited} />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
