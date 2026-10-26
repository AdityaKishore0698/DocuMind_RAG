# ResearchLens — Frontend

Next.js 16 (App Router) + TypeScript + Tailwind CSS v4, a client-side single-page app. It signs users in with
Supabase Auth and calls the FastAPI backend with the Supabase access token as a Bearer token.

## Setup

Follow the repository [README](../README.md#quick-start-local-no-api-keys): `scripts/setup_local_env.sh`
writes `frontend/.env.local` from the local Supabase stack. Then:

```bash
npm ci
npm run dev        # http://localhost:3000
```

The backend's `FRONTEND_ORIGINS` must include this origin (default `http://localhost:3000`).
[`.env.example`](.env.example) documents the variables.

## Layout

| Path | Purpose |
| --- | --- |
| `app/layout.tsx`, `app/page.tsx` | Root shell; renders the sign-in view or the app |
| `lib/api.ts` | Typed API client: documents, chat streaming with sources, comparison, gaps |
| `lib/auth-context.tsx`, `lib/supabase.ts` | Supabase session, sign-up/in, 401 → sign-out |
| `components/LoginView.tsx` | Sign in / register / reset (Google button only with `NEXT_PUBLIC_ENABLE_GOOGLE_AUTH=true`) |
| `components/AppShell.tsx` | Layout, paper selection, chat vs. comparison view |
| `components/Sidebar.tsx` | Upload, paper list (select, re-process), chat history, account |
| `components/ChatPanel.tsx` | Streaming chat, cited-source list, Deep-analysis toggle |
| `components/ComparePanel.tsx` | Comparison table and research gaps |
| `components/Sources.tsx` | Citation chips and source lists |

## Streaming

`POST /chat/chat` returns a chunked text body plus the `X-Session-ID`, `X-Chat-Mode` and `X-Sources` headers
(base64url JSON of the numbered sources). `streamChat` reads the body with `response.body.getReader()` and emits
text deltas; the sources arrive before the first token.

## Scripts

```bash
npm run dev     # dev server (Turbopack)
npm run build   # production build
npm run start   # serve the production build
npm run lint    # eslint
```
