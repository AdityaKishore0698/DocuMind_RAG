#!/usr/bin/env bash
#
# Single-container entrypoint for Koyeb Free Eco (1 service, 512 MB RAM):
#   - Celery worker  (background, --pool=solo to avoid prefork child processes)
#   - FastAPI / uvicorn (foreground-supervised, 1 worker)
#
# Both processes are backgrounded and supervised so that (a) SIGTERM/SIGINT from
# the platform shuts both down gracefully and (b) if either dies the container
# exits and Koyeb restarts it.
#
set -uo pipefail

export PYTHONUNBUFFERED=1

# api/ and worker/ keep their original flat imports (from core.database import ...,
# from celery_app import ..., etc). Put each directory on PYTHONPATH so the module
# entrypoints resolve when launched from the repo root.
echo "start.sh: starting Celery worker"
PYTHONPATH=/app/worker celery -A celery_app worker --pool=solo --loglevel=info &
worker_pid=$!

echo "start.sh: starting FastAPI (uvicorn) on :8000"
PYTHONPATH=/app/api uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
api_pid=$!

_shutting_down=0
shutdown() {
    [ "$_shutting_down" = 1 ] && return
    _shutting_down=1
    trap - TERM INT
    echo "start.sh: terminating worker ($worker_pid) and api ($api_pid)"
    kill -TERM "$worker_pid" "$api_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
    echo "start.sh: shutdown complete"
}
trap shutdown TERM INT

# Block until either child exits, then take the whole container down.
wait -n
echo "start.sh: a child process exited — shutting down the container"
shutdown
exit 1
