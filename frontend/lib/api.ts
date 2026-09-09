import { API_URL } from "./config";
import { supabase } from "./supabase";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export interface DocumentItem {
  id: number;
  filename: string;
  status: string; // uploaded | processing | completed | failed
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
}

export interface ChatMessageItem {
  role: "user" | "assistant";
  content: string;
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

export function listDocuments(token: string): Promise<DocumentItem[]> {
  // Trailing slash matters — avoids a 307 that would drop the Auth header.
  return jsonRequest<DocumentItem[]>("/document/", {
    headers: authHeaders(token),
    cache: "no-store",
  });
}

export async function uploadDocuments(
  token: string,
  files: File[],
): Promise<UploadResult[]> {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  const data = await jsonRequest<{ uploaded_files: UploadResult[] }>(
    "/document/upload",
    { method: "POST", headers: authHeaders(token), body: form },
  );
  return data.uploaded_files;
}

export function deleteDocument(token: string, id: number): Promise<void> {
  return jsonRequest(`/document/${id}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
}

/* ------------------------------------------------------------------ */
/* Chat sessions                                                       */
/* ------------------------------------------------------------------ */

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

export function deleteSession(token: string, sessionId: number): Promise<void> {
  return jsonRequest(`/chat/sessions/${sessionId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
}

/* ------------------------------------------------------------------ */
/* Chat streaming                                                      */
/* ------------------------------------------------------------------ */

export interface StreamHandlers {
  onToken: (chunk: string) => void;
  signal?: AbortSignal;
}

export interface StreamOutcome {
  text: string;
  sessionId: number | null;
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
  payload: { query: string; session_id?: number | null },
  { onToken, signal }: StreamHandlers,
): Promise<StreamOutcome> {
  const res = await fetch(`${API_URL}/chat/chat`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({
      query: payload.query,
      ...(payload.session_id ? { session_id: payload.session_id } : {}),
    }),
    signal,
  });

  if (!res.ok) throw await toError(res);

  const header = res.headers.get("X-Session-ID");
  const sessionId = header ? Number(header) : null;

  if (!res.body) {
    const text = await res.text();
    if (text) onToken(text);
    return { text, sessionId };
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

  return { text, sessionId };
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
