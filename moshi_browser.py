"""Live Moshi session over a WebSocket, for the browser.

The offline milestones drove Moshi from a fixed fixture. This drives it from a
live microphone instead. The loop is the same one already measured: Mimi encodes
1920 samples into 8 codebooks, LMGen advances one frame, Mimi decodes back to
audio.

Transport is raw int16 PCM at 24 kHz in 1920-sample frames, which is 80 ms, the
model's native cadence. Opus would buy bandwidth we do not need at 48 KB/s and
would add a codec to both ends for no measurement benefit.

Three details are borrowed from Kyutai's own server because they are the right
answers and we would otherwise have had to rediscover them:

  - one byte of message kind in front of every binary frame,
  - a warm-up of silent frames through the full path before serving, which is
    the same trick the offline milestone arrived at independently,
  - the first inbound frame is discarded, because from the model's point of view
    it is already in the past.

Streaming state is started once and reset per connection, so the CUDA graph
capture paid at startup is not paid again for every session.

An open connection keeps this container alive and billing, silence included.
That is accepted and measured rather than papered over: the cost of an idle
session is itself one of the questions the article asks.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

from moshi_experiments import DATA_ROOT, EXPLICIT_WARMUP_FRAMES, _resolve_weights

# One byte of kind in front of every binary frame.
KIND_PCM = 1
KIND_JSON = 2

SESSION_LOG_DIR = f"{DATA_ROOT}/outputs/browser"

# Where modal_app.live_image mounts the page inside the container. The two must
# agree; if they do not, the first request fails on the path rather than
# quietly serving nothing.
PAGE_PATH = "/root/browser/index.html"


def _noop() -> None:
    """Default durability hook. A plain filesystem needs nothing."""


def _emit(record: dict) -> None:
    """Print one JSON object per line, so it lands in the platform log."""
    print(json.dumps(record), flush=True)


class SessionLog:
    """Append-only event log for one connection.

    Times are this process's monotonic clock, measured from the moment the
    socket was accepted. The browser has its own clock and we have not measured
    the offset between them, so browser timestamps are recorded exactly as
    reported and are never subtracted from these.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.path = Path(SESSION_LOG_DIR) / f"session-{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a")
        self._start = time.perf_counter()
        self._first_seen: set[str] = set()

    def event(self, name: str, **fields) -> None:
        record = {
            "event": name,
            "session": self.session_id,
            "server_ms": round((time.perf_counter() - self._start) * 1000, 1),
            **fields,
        }
        _emit(record)
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def first(self, name: str, **fields) -> None:
        """Emit only the first occurrence, for one-shot milestone events."""
        if name in self._first_seen:
            return
        self._first_seen.add(name)
        self.event(name, **fields)

    def close(self) -> None:
        self._handle.close()


class Runtime:
    """The loaded model, shared by every connection this container serves."""

    def __init__(self, mimi, lm_gen, torch, commit):
        self.mimi = mimi
        self.lm_gen = lm_gen
        self.torch = torch
        self.commit = commit
        self.frame_size = mimi.frame_size
        self.lock = asyncio.Lock()

    def handle_frame(self, chunk):
        """One 80 ms frame in, decoded audio out, or None while the model waits."""
        codes = self.mimi.encode(chunk)
        tokens = self.lm_gen.step(codes)
        if tokens is None:
            return None
        audio = self.mimi.decode(tokens[:, 1:])
        return (
            (audio[0, 0].clamp(-1, 1) * 32767).to(self.torch.int16).cpu().numpy().tobytes()
        )


def _build(commit) -> Runtime:
    """Load the model, warm it up, and start streaming state once."""
    import torch
    from moshi.models import LMGen, loaders

    started = time.perf_counter()
    weights = _resolve_weights(loaders)
    mimi = loaders.get_mimi(weights["mimi"], device="cuda")
    lm_gen = LMGen(
        loaders.get_moshi_lm(weights["moshi"], device="cuda"),
        temp=0.8,
        temp_text=0.7,
    )
    torch.cuda.synchronize()

    # Started once for the life of the container. Each connection resets the
    # state but keeps the graph wrappers this allocated.
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    silence = torch.zeros(1, 1, mimi.frame_size, dtype=torch.float32, device="cuda")
    warmup_start = time.perf_counter()
    for _ in range(EXPLICIT_WARMUP_FRAMES):
        codes = mimi.encode(silence)
        tokens = lm_gen.step(codes)
        if tokens is not None:
            mimi.decode(tokens[:, 1:])
    torch.cuda.synchronize()
    warmup_ms = round((time.perf_counter() - warmup_start) * 1000)

    mimi.reset_streaming()
    lm_gen.reset_streaming()

    _emit(
        {
            "event": "server_ready",
            "load_and_warmup_ms": round((time.perf_counter() - started) * 1000),
            "warmup_ms": warmup_ms,
            "warmup_frames": EXPLICIT_WARMUP_FRAMES,
            "frame_size": mimi.frame_size,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
    )
    return Runtime(mimi, lm_gen, torch, commit)


async def _serve(sock, runtime) -> None:
    import numpy as np

    log = SessionLog(uuid.uuid4().hex[:8])
    try:
        await sock.accept()
        log.event("ws_connected", frame_size=runtime.frame_size)
        await sock.send_bytes(
            bytes([KIND_JSON])
            + json.dumps(
                {
                    "event": "handshake",
                    "session": log.session_id,
                    "frame_size": runtime.frame_size,
                    "kind_pcm": KIND_PCM,
                    "kind_json": KIND_JSON,
                }
            ).encode()
        )

        # One conversation at a time, matching the official server: there is a
        # single streaming state and a single GPU behind it.
        async with runtime.lock:
            runtime.mimi.reset_streaming()
            runtime.lm_gen.reset_streaming()

            pending = np.zeros(0, dtype=np.float32)
            skip_frames = 1

            while True:
                message = await sock.receive_bytes()
                if not message:
                    continue
                kind, payload = message[0], message[1:]

                if kind == KIND_JSON:
                    try:
                        client_event = json.loads(payload)
                    except ValueError:
                        continue
                    log.event(
                        "client_" + str(client_event.get("event", "unknown")),
                        browser_ms=client_event.get("t"),
                    )
                    continue

                if kind != KIND_PCM:
                    log.event("unknown_kind", kind=kind, length=len(payload))
                    continue

                log.first("first_mic_audio", bytes=len(payload))
                samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                pending = np.concatenate((pending, samples))

                while pending.shape[-1] >= runtime.frame_size:
                    frame = pending[: runtime.frame_size]
                    pending = pending[runtime.frame_size :]

                    if skip_frames:
                        # The model sees the first frame as already in the past.
                        # Encode it anyway, then reset, so the next call gets the
                        # padding the encoder expects at the start.
                        runtime.mimi.encode(
                            runtime.torch.from_numpy(frame.copy()).to(device="cuda")[None, None]
                        )
                        runtime.mimi.reset_streaming()
                        skip_frames -= 1
                        continue

                    started = time.perf_counter()
                    chunk = runtime.torch.from_numpy(frame).to(device="cuda")[None, None]
                    outbound = runtime.handle_frame(chunk)
                    elapsed = (time.perf_counter() - started) * 1000

                    if outbound:
                        log.first("first_model_step")
                        log.first("first_model_audio")
                        log.first("first_chunk_sent", frame_ms=round(elapsed, 1))
                        await sock.send_bytes(bytes([KIND_PCM]) + outbound)
    except Exception as exc:
        log.event("disconnect", reason=type(exc).__name__, detail=str(exc)[:200])
    finally:
        try:
            runtime.mimi.reset_streaming()
            runtime.lm_gen.reset_streaming()
        except Exception:
            pass
        log.event("session_end")
        log.close()
        # The event log has to outlive the connection, so the provider is asked
        # to make it durable.
        runtime.commit()


def create_app(commit=_noop):
    """Build the ASGI app. Expected to run once per container, not per request."""
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse

    runtime = _build(commit)
    page = Path(PAGE_PATH).read_text()

    app = FastAPI()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "frame_size": runtime.frame_size}

    @app.websocket("/ws")
    async def websocket_endpoint(sock: WebSocket) -> None:
        await _serve(sock, runtime)

    return app
