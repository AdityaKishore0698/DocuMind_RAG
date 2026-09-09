/**
 * Base URL of the FastAPI backend. Override per-environment with
 * NEXT_PUBLIC_API_URL (see .env.example). Falls back to the local dev API.
 */
export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
