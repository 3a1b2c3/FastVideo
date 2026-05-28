# pyright: reportArgumentType=false, reportMissingImports=false
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastvideo.entrypoints.streaming import build_health_router
from dreamverse.gpu_pool import GPUPool, get_available_gpus
from dreamverse.session_logger import SessionEventLogger

from dreamverse.config import (
    DEVTOOLS_ENABLED,
    FRONTEND_STATIC_DIR_CANDIDATES,
    PROMPT_SAFETY_ENABLED,
    SESSION_LOG_ROOT,
)
from dreamverse.prompt_enhancer import PromptEnhancer
from dreamverse.prompt_safety import PromptSafetyFilter

import dreamverse.runtime as runtime
from dreamverse.routes.health import (
    router as internal_monitor_router, )
from dreamverse.routes.presets import (
    prompt_config_router,
    curated_presets_router,
)
from dreamverse.session.controller import SessionController


class _HeartbeatAccessLogFilter(logging.Filter):
    """Drop noisy access logs for frequent health/readiness probes."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return ('"GET /healthz ' not in message and '"GET /readyz ' not in message)


def _install_heartbeat_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    for existing in access_logger.filters:
        if isinstance(existing, _HeartbeatAccessLogFilter):
            return
    access_logger.addFilter(_HeartbeatAccessLogFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("Starting server...")

    # Get available GPUs
    gpu_ids = get_available_gpus()
    print(f"Selected GPU ids: {gpu_ids}")

    # Initialize GPU pool (spawns subprocess per GPU)
    runtime.gpu_pool = GPUPool(gpu_ids)
    await runtime.gpu_pool.initialize()

    runtime.prompt_enhancer = PromptEnhancer()
    runtime.session_event_logger = SessionEventLogger(Path(SESSION_LOG_ROOT))
    runtime.prompt_safety_filter = (PromptSafetyFilter() if PROMPT_SAFETY_ENABLED else None)
    if runtime.prompt_safety_filter is not None:
        print("Prompt safety filter enabled")

    print("Server started")
    yield

    print("Shutting down server...")
    await runtime.gpu_pool.shutdown()
    runtime.prompt_safety_filter = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(build_health_router(lambda: runtime.gpu_pool))
app.include_router(internal_monitor_router)
app.include_router(prompt_config_router)
if DEVTOOLS_ENABLED:
    app.include_router(curated_presets_router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    controller = SessionController(
        ws=websocket,
        gpu_pool=runtime.gpu_pool,
        prompt_enhancer=runtime.prompt_enhancer,
        prompt_safety_filter=runtime.prompt_safety_filter,
        session_event_logger=runtime.session_event_logger,
    )
    await controller.run()


# Serve an exported frontend bundle when present.
for static_dir in FRONTEND_STATIC_DIR_CANDIDATES:
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        break


def cli() -> None:
    import argparse
    import sys
    import uvicorn
    from pathlib import Path

    from dreamverse._deps import require_dreamverse_runtime_deps

    require_dreamverse_runtime_deps()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--log-file",
                        default=str(Path(__file__).resolve().parent.parent / "outputs" / "backend.log"),
                        help="Tee server stdout/stderr to this file (default: "
                             "<repo>/apps/dreamverse/outputs/backend.log). "
                             "Set to empty string to disable.")
    args = parser.parse_args()

    # Tee backend stdout/stderr to a log file. Companion to gpu_pool.py's
    # per-worker tee — the worker writes to gpu_<id>_worker.log; this writes
    # parent (uvicorn + dreamverse + GPU pool orchestrator) output to
    # backend.log. Survives the Windows subprocess _readerthread races that
    # otherwise lose buffered output to the terminal.
    if args.log_file:
        _log_path = Path(args.log_file)
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(_log_path, "w", buffering=1, encoding="utf-8", errors="replace")
        _log_file.write(f"# dreamverse-server pid={os.getpid()} host={args.host} port={args.port}\n")
        _log_file.flush()

        class _TeeStream:
            """Write to underlying stream AND log file. Best-effort: never raise."""
            def __init__(self, primary, secondary):
                self._primary = primary
                self._secondary = secondary
            def write(self, s):
                try:
                    self._primary.write(s)
                except Exception:
                    pass
                try:
                    self._secondary.write(s)
                    self._secondary.flush()
                except Exception:
                    pass
                return len(s) if s else 0
            def flush(self):
                for stream in (self._primary, self._secondary):
                    try:
                        stream.flush()
                    except Exception:
                        pass
            def __getattr__(self, name):
                return getattr(self._primary, name)

        sys.stdout = _TeeStream(sys.stdout, _log_file)
        sys.stderr = _TeeStream(sys.stderr, _log_file)
        print(f"[dreamverse-server] backend logging to: {_log_path}")

    _install_heartbeat_log_filter()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
