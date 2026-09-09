# DocuMind — Frontend

Next.js 16 (App Router) + TypeScript + Tailwind CSS. A single-page client app that talks
directly to the FastAPI backend.

## Setup

```bash
npm install
cp .env.example .env.local     # then edit NEXT_PUBLIC_API_URL if the API isn't on :8000
npm run dev                    # http://localhost:3000
```

The backend must be running and its `FRONTEND_ORIGINS` must include this app's origin
(`http://localhost:3000` by default) or the browser will block the requests.

## Layout

| Path | Purpose |
| --- | --- |
| `app/layout.tsx`, `app/page.tsx` | Root shell; renders the login view or the app |
| `lib/config.ts` | `API_URL`, storage keys |
| `lib/api.ts` | Typed API client + `streamChat` (Streams-API reader) |
| `lib/auth-context.tsx` | `useAuth()` — JWT in `localStorage`, 401 → auto sign-out |
| `components/LoginView.tsx` | Sign in / register |
| `components/AppShell.tsx` | Two-pane layout, session + message state |
| `components/Sidebar.tsx` | Left pane — document upload/list, chat history, account |
| `components/ChatPanel.tsx` | Right pane — streaming chat, stop button |

## Streaming

`POST /chat/chat` returns a chunked response body (and an `X-Session-ID` header).
`streamChat` in `lib/api.ts` reads it with `response.body.getReader()` + `TextDecoder`
and emits text deltas via `onToken`. It also unwraps SSE `data:` framing, so it keeps
working if the endpoint is later switched to `text/event-stream`.

## Scripts

```bash
npm run dev        # dev server (Turbopack)
npm run build      # production build
npm run start      # serve the production build
npm run lint       # eslint
```
