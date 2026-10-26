import { API_URL } from "./config";
import { supabase } from "./supabase";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export interface DocumentItem {
  id: number;
  /** The chat the paper belongs to; null for papers uploaded before papers were chat-scoped. */
  session_id?: number | null;
  filename: string;
  status: string; // uploaded | processing | completed | failed
  title?: string | null;
  authors?: string | null;
  year?: number | null;
  page_count?: number | null;
  error?: string | null;
  sections?: string[];
  /** Embedding space of the stored vectors, e.g. "ollama:nomic-embed-text". */
  embedding_model?: string | null;
  /** False when the paper was embedded with a different model than the server uses: re-process it. */
  searchable?: boolean;
}

/** A numbered source the answer can cite as [S n]. */
export interface SourceItem {
  n: number;
  chunk_id: number;
  document_id: number;
  title: string | null;
  filename: string | null;
  page_number: number | null;
  section: string | null;
  snippet: string;
}

export interface MessageSources {
  sources: SourceItem[];
  cited: number[];
  invalid: number[];
  mode: ChatMode;
  failed?: boolean;
}

export type ChatMode = "rag" | "agent";

export interface Capabilities {
  agent: boolean;
  retrieval_mode: string;
  top_k: number;
  max_selected: number;
  embedding_space: string;
}

export interface UploadResult {
  task_id: string;
  document_id: number;
  filename: string;
}

export interface ChatSessionItem {
  id: number;
  title: string;
  created_at?: string;
  /** Papers uploaded in this chat (they are deleted with it). */
  paper_count?: number;
}

export interface ChatMessageItem {
  role: "user" | "assistant";
  content: string;
  sources?: MessageSources | null;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message || `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
  }
}

/* ------------------------------------------------------------------ */
/* Low-level helpers                                                   */
/* ------------------------------------------------------------------ */

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

async function toError(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body?.detail ?? JSON.stringify(body);
  } catch {
    try {
      detail = (await res.text()) || detail;
    } catch {
      /* ignore */
    }
  }
  // A dead/expired token — sign out so the app returns to the login screen.
  if (res.status === 401 && typeof window !== "undefined") {
    void supabase.auth.signOut();
  }
  return new ApiError(res.status, detail);
}

async function jsonRequest<T>(
  path: string,
  init: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) throw await toError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/* ------------------------------------------------------------------ */
/* Auth                                                                */
/* ------------------------------------------------------------------ */

export async function register(
  username: string,
  email: string,
  password: string,
): Promise<void> {
  await jsonRequest("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  });
}

export async function login(
  username: string,
  password: string,
): Promise<string> {
  // OAuth2 password flow — the backend expects form-encoded data.
  const form = new URLSearchParams({ username, password });
  const data = await jsonRequest<{ access_token: string }>("/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  return data.access_token;
}

export async function deleteAccount(token: string): Promise<void> {
  await jsonRequest("/auth/profile", {
    method: "DELETE",
    headers: authHeaders(token),
  });
}

/* ------------------------------------------------------------------ */
/* Documents                                                           */
/* ------------------------------------------------------------------ */

// Papers are chat-scoped: every call names the chat, and the API rejects papers of other chats.

export function listDocuments(token: string, sessionId: number): Promise<DocumentItem[]> {
  // Trailing slash matters — avoids a 307 that would drop the Auth header.
  return jsonRequest<DocumentItem[]>(`/document/?session_id=${sessionId}`, {
    headers: authHeaders(token),
    cache: "no-store",
  });
}

/** Papers uploaded before papers belonged to chats; searched nowhere until moved into a chat. */
export function listUnassignedDocuments(token: string): Promise<DocumentItem[]> {
  return jsonRequest<DocumentItem[]>("/document/unassigned", {
    headers: authHeaders(token),
    cache: "no-store",
  });
}

export function assignDocument(token: string, id: number, sessionId: number): Promise<DocumentItem> {
  return jsonRequest<DocumentItem>(`/document/${id}/assign?session_id=${sessionId}`, {
    method: "POST",
    headers: authHeaders(token),
  });
}

export async function uploadDocuments(
  token: string,
  sessionId: number,
  files: File[],
): Promise<UploadResult[]> {
  const form = new FormData();
  form.append("session_id", String(sessionId));
  for (const file of files) form.append("files", file, file.name);
  const data = await jsonRequest<{ uploaded_files: UploadResult[] }>(
    "/document/upload",
    { method: "POST", headers: authHeaders(token), body: form },
  );
  return data.uploaded_files;
}

export function reprocessDocument(
  token: string,
  id: number,
  sessionId: number,
): Promise<{ task_id: string }> {
  return jsonRequest(`/document/${id}/reprocess?session_id=${sessionId}`, {
    method: "POST",
    headers: authHeaders(token),
  });
}

/** `sessionId` null deletes an unassigned (pre-chat) paper. */
export function deleteDocument(
  token: string,
  id: number,
  sessionId: number | null,
): Promise<DeleteResult> {
  const query = sessionId === null ? "" : `?session_id=${sessionId}`;
  return jsonRequest(`/document/${id}${query}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
}

/* ------------------------------------------------------------------ */
/* Chat sessions                                                       */
/* ------------------------------------------------------------------ */

export function createSession(token: string, title: string): Promise<ChatSessionItem> {
  return jsonRequest<ChatSessionItem>("/chat/sessions", {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export function listSessions(token: string): Promise<ChatSessionItem[]> {
  return jsonRequest<ChatSessionItem[]>("/chat/sessions", {
    headers: authHeaders(token),
    cache: "no-store",
  });
}

export function getSessionMessages(
  token: string,
  sessionId: number,
): Promise<ChatMessageItem[]> {
  return jsonRequest<ChatMessageItem[]>(`/chat/sessions/${sessionId}/messages`, {
    headers: authHeaders(token),
    cache: "no-store",
  });
}

export function renameSession(
  token: string,
  sessionId: number,
  title: string,
): Promise<ChatSessionItem> {
  return jsonRequest<ChatSessionItem>(`/chat/sessions/${sessionId}`, {
    method: "PUT",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export interface DeleteResult {
  papers_deleted?: number;
  /** Stored files that could not be removed (the database rows are already gone). */
  files_not_removed?: number;
}

/** Deletes the chat, its history, its papers and their stored files. */
export function deleteSession(token: string, sessionId: number): Promise<DeleteResult> {
  return jsonRequest(`/chat/sessions/${sessionId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
}

export function getCapabilities(token: string): Promise<Capabilities> {
  return jsonRequest<Capabilities>("/chat/capabilities", {
    headers: authHeaders(token),
    cache: "no-store",
  });
}

/* ------------------------------------------------------------------ */
/* Multi-paper analysis                                                */
/* ------------------------------------------------------------------ */

export type CompareAspect = "profile" | "methods" | "results" | "limitations";

export interface CitedValue {
  value: string;
  citations: number[];
}

export interface ComparedPaper {
  document_id: number;
  title: string;
  filename: string;
  authors: string | null;
  year: number | null;
  fields: Record<string, CitedValue>;
  sources: SourceItem[];
  warnings: string[];
  error: string | null;
}

export interface CompareResult {
  aspect: CompareAspect;
  fields: Record<string, string>;
  papers: ComparedPaper[];
  warnings: string[];
}

export interface GapItem {
  gap: string;
  rationale: string;
  citations: number[];
  /** Decided by the server from the citations: ≥ 2 distinct cited papers = cross-paper. */
  label: "cross_paper_hypothesis" | "single_paper_observation";
  papers: number[];
}

export interface GapsResult {
  papers: { document_id: number; title: string; letter: string }[];
  gaps: GapItem[];
  sources: SourceItem[];
  disclaimer: string;
  warnings: string[];
}

export function comparePapers(
  token: string,
  sessionId: number,
  documentIds: number[],
  aspect: CompareAspect,
  signal?: AbortSignal,
): Promise<CompareResult> {
  return jsonRequest<CompareResult>("/analysis/compare", {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, document_ids: documentIds, aspect }),
    signal,
  });
}

export function findGaps(
  token: string,
  sessionId: number,
  documentIds: number[],
  signal?: AbortSignal,
): Promise<GapsResult> {
  return jsonRequest<GapsResult>("/analysis/gaps", {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, document_ids: documentIds }),
    signal,
  });
}

/* ------------------------------------------------------------------ */
/* Chat streaming                                                      */
/* ------------------------------------------------------------------ */

export interface StreamHandlers {
  onToken: (chunk: string) => void;
  /** Called once, before the first token, with the sources the answer may cite. */
  onSources?: (sources: SourceItem[], mode: ChatMode) => void;
  signal?: AbortSignal;
}

export interface StreamOutcome {
  text: string;
  sessionId: number | null;
  sources: SourceItem[];
  mode: ChatMode;
}

export interface ChatPayload {
  query: string;
  session_id?: number | null;
  document_ids?: number[];
  mode?: ChatMode;
}

/** Decode the base64url JSON `X-Sources` header. */
function decodeSources(header: string | null): SourceItem[] {
  if (!header) return [];
  try {
    const b64 = header.replace(/-/g, "+").replace(/_/g, "/");
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const parsed = JSON.parse(new TextDecoder().decode(bytes));
    return Array.isArray(parsed) ? (parsed as SourceItem[]) : [];
  } catch {
    return [];
  }
}

/** Which of `sources` the answer text cites ([S1], [S2, S3] …). */
export function citedNumbers(text: string, count: number): number[] {
  const found = new Set<number>();
  for (const group of text.matchAll(/\[([^\[\]]{1,40})\]/g)) {
    for (const m of group[1].matchAll(/S\s*(\d+)/gi)) {
      const n = Number(m[1]);
      if (n >= 1 && n <= count) found.add(n);
    }
  }
  return [...found].sort((a, b) => a - b);
}

/**
 * POST /chat/chat and consume the streamed response incrementally.
 *
 * The backend streams the answer as a chunked body (labelled `text/plain`);
 * this reader also tolerates formal SSE framing (`data:` lines) in case the
 * endpoint is upgraded to `text/event-stream` later. Either way, `onToken`
 * is called with plain text deltas as they arrive.
 */
export async function streamChat(
  token: string,
  payload: ChatPayload,
  { onToken, onSources, signal }: StreamHandlers,
): Promise<StreamOutcome> {
  const res = await fetch(`${API_URL}/chat/chat`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({
      query: payload.query,
      ...(payload.session_id ? { session_id: payload.session_id } : {}),
      ...(payload.document_ids?.length ? { document_ids: payload.document_ids } : {}),
      ...(payload.mode ? { mode: payload.mode } : {}),
    }),
    signal,
  });

  if (!res.ok) throw await toError(res);

  const header = res.headers.get("X-Session-ID");
  const sessionId = header ? Number(header) : null;
  const sources = decodeSources(res.headers.get("X-Sources"));
  const mode: ChatMode = res.headers.get("X-Chat-Mode") === "agent" ? "agent" : "rag";
  onSources?.(sources, mode);

  if (!res.body) {
    const text = await res.text();
    if (text) onToken(text);
    return { text, sessionId, sources, mode };
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let text = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const raw = decoder.decode(value, { stream: true });
      const delta = parseChunk(raw);
      if (delta) {
        text += delta;
        onToken(delta);
      }
    }
  } finally {
    reader.releaseLock();
  }

  return { text, sessionId, sources, mode };
}

/** Strip SSE `data:` framing if present; otherwise pass the text through. */
function parseChunk(raw: string): string {
  if (!/^\s*(data|event|id|retry):/m.test(raw)) return raw;
  return raw
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .filter((line) => line !== "[DONE]")
    .join("");
}
