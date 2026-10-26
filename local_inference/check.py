"""Staged checks for the local-inference path, cheapest first.

    python local_inference/check.py                # config + server version + model metadata (no model load)
    python local_inference/check.py --embed        # + one embedding (loads nomic-embed-text, ~0.3 GB)
    python local_inference/check.py --generate     # + one short generation (loads OLLAMA_CHAT_MODEL)
    python local_inference/check.py --rag          # + synthetic ingest/retrieve/answer in an isolated DB

Uses the application's own provider (api/core/inference.py). Prints no prompts,
document text or secrets. Run the inference stages only after checking GPU headroom
on the shared lab server (local_inference/README.md); the first stage loads nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import httpx  # noqa: E402

from core import inference  # noqa: E402

# Synthetic, non-sensitive corpus for the --rag stage.
SYNTHETIC_DOCS = [
    "The Zephyr gripper uses three silicone fingers and was tested on 40 household objects.",
    "Policy training for the Zephyr gripper took 12 hours on a single GPU with a batch size of 64.",
    "The warehouse robot Atlas-7 navigates using lidar and reached 98 percent pick accuracy.",
]
RAG_QUESTION = "How long did Zephyr gripper policy training take?"


def _get(settings, path: str, **kw):
    with httpx.Client(timeout=httpx.Timeout(10, connect=settings.connect_timeout_s), trust_env=False) as c:
        return c.request("POST" if kw else "GET", settings.base_url + path, **kw)


def stage_metadata(settings) -> dict:
    out = {"base_url": settings.base_url, "chat_model": settings.chat_model,
           "embedding_model": settings.embedding_model}
    out["server_version"] = _get(settings, "/api/version").json().get("version")
    out["loaded_models_before"] = [m.get("name") for m in _get(settings, "/api/ps").json().get("models", [])]
    for role, name in (("embedding", settings.embedding_model), ("chat", settings.chat_model)):
        response = _get(settings, "/api/show", json={"model": name})
        if response.status_code != 200:
            out[role] = {"available": False, "error": response.json().get("error")}
            continue
        info, details = response.json().get("model_info", {}), response.json().get("details", {})
        out[role] = {"available": True, "parameter_size": details.get("parameter_size"),
                     "quantization": details.get("quantization_level"),
                     **{k.split(".", 1)[1]: v for k, v in info.items()
                        if k.endswith(("embedding_length", "context_length"))}}
    return out


def stage_embed(provider) -> dict:
    t0 = time.perf_counter()
    vec = provider.embed("A short synthetic sentence about robot grippers.", kind="document")
    return {"dims": len(vec), "all_finite": all(math.isfinite(x) for x in vec),
            "l2_norm": round(math.sqrt(sum(x * x for x in vec)), 4), "seconds": round(time.perf_counter() - t0, 2)}


async def _stream(provider, system, question) -> dict:
    t0, ttft, text = time.perf_counter(), None, ""
    async for piece in provider.stream_chat(system, [], question):
        ttft = ttft if ttft is not None else time.perf_counter() - t0
        text += piece
    return {"ttft_s": round(ttft or 0, 2), "total_s": round(time.perf_counter() - t0, 2), "chars": len(text),
            "answer_preview": text.strip()[:160]}


def stage_rag(provider, database_url: str) -> dict:
    import psycopg2

    host, name = urlparse(database_url).hostname, urlparse(database_url).path.lstrip("/")
    if host not in inference.LOOPBACK_HOSTS or "local" not in name or "eval" in name:
        raise SystemExit("--rag needs LOCAL_INFERENCE_DATABASE_URL on localhost whose name contains 'local' "
                         "(and not the Gemini evaluation DB)")
    def vec(values) -> str:  # pgvector text literal; cast with ::vector (no numpy dependency)
        return "[" + ",".join(repr(float(x)) for x in values) + "]"

    conn = psycopg2.connect(database_url)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    with conn.cursor() as cur:  # a private table in the isolated DB; nothing else is touched
        cur.execute("CREATE TABLE IF NOT EXISTS smoke_chunks (id serial PRIMARY KEY, content text, "
                    "embedding vector(768))")
        cur.execute("TRUNCATE smoke_chunks")
        for doc in SYNTHETIC_DOCS:
            cur.execute("INSERT INTO smoke_chunks (content, embedding) VALUES (%s, %s::vector)",
                        (doc, vec(provider.embed(doc, kind="document"))))
        q = vec(provider.embed(RAG_QUESTION, kind="query"))
        cur.execute("SELECT id, content, embedding <=> %s::vector AS d FROM smoke_chunks "
                    "ORDER BY embedding <=> %s::vector LIMIT 2", (q, q))
        rows = cur.fetchall()
    conn.commit()
    conn.close()
    context = "\n\n".join(r[1] for r in rows)
    system = ("You are ResearchLens, an assistant that answers questions using the user's uploaded documents. "
              "Rely on the context below; if it does not contain the answer, say so.\n\n"
              f"Context from documents:\n{context}")
    gen = asyncio.run(_stream(provider, system, RAG_QUESTION))
    return {"top_ids": [r[0] for r in rows], "top_distances": [round(r[2], 4) for r in rows],
            "expected_top_id": 2, "generation": gen}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--rag", action="store_true")
    args = parser.parse_args()

    settings = inference.ollama_settings()
    report = {"metadata": stage_metadata(settings)}
    if args.embed or args.generate or args.rag:
        provider = inference.OllamaProvider(settings)
        if args.embed:
            report["embed"] = stage_embed(provider)
        if args.generate:
            report["generate"] = asyncio.run(_stream(provider, "Answer in one short sentence.",
                                                     "What is a robot gripper?"))
        if args.rag:
            report["rag"] = stage_rag(provider, os.environ.get("LOCAL_INFERENCE_DATABASE_URL", ""))
        report["loaded_models_after"] = [
            {"name": m.get("name"), "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 2)}
            for m in _get(settings, "/api/ps").json().get("models", [])]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
