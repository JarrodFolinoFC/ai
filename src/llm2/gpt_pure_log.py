"""
Detailed logging for gpt_pure.py — ships events to OpenObserve.

Off by default. Two ways to turn it on:

  1. Env vars (no code changes):
       GPT_PURE_LOG=1                  # enable
       GPT_PURE_LOG_SAMPLE=0.05        # 5% sample of inner events (default 0.01)
       GPT_PURE_LOG_URL=http://localhost:5080/api/default/gpt_pure/_json
       OO_USER=root@example.com        # optional basic auth
       OO_PASSWORD=...

  2. Programmatic:
       import gpt_pure_log
       gpt_pure_log.enable_logging(sample_rate=0.05)

The module exposes one logger ("gpt_pure") plus a `should_sample()` helper
that the gpt_pure code uses to gate high-volume events. Always-on events
(train_step, generate_token) call the logger directly.
"""

import atexit
import json
import logging
import os
import random
import threading
from urllib import request as urlrequest
from urllib.error import URLError

LOGGER_NAME = "gpt_pure"
DEFAULT_URL = "http://localhost:5080/api/default/gpt_pure/_json"
DEFAULT_SAMPLE_RATE = 0.01
DEFAULT_BATCH_SIZE = 50

_state = {"sample_rate": 0.0, "enabled": False}


def should_sample():
    """Cheap gate for high-volume call sites. Returns False when logging disabled."""
    if not _state["enabled"]:
        return False
    return random.random() < _state["sample_rate"]


class OpenObserveHandler(logging.Handler):
    """Batches log records as JSON and POSTs to OpenObserve's _json ingest API."""

    def __init__(self, url, batch_size=DEFAULT_BATCH_SIZE, auth=None):
        super().__init__()
        self.url = url
        self.batch_size = batch_size
        self.auth = auth  # (user, password) or None
        self._buf = []
        self._lock = threading.Lock()

    def emit(self, record):
        try:
            payload = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = getattr(record, "fields", None)
            if isinstance(extra, dict):
                payload.update(extra)
            with self._lock:
                self._buf.append(payload)
                if len(self._buf) >= self.batch_size:
                    self._flush_locked()
        except Exception:
            self.handleError(record)

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buf:
            return
        body = json.dumps(self._buf).encode("utf-8")
        self._buf = []
        req = urlrequest.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.auth:
            import base64

            token = base64.b64encode(f"{self.auth[0]}:{self.auth[1]}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        try:
            urlrequest.urlopen(req, timeout=2.0).close()
        except (URLError, OSError):
            # Don't let logging failures crash training.
            pass


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def log_event(event, **fields):
    """Structured event helper. Adds `event` to the JSON payload."""
    if not _state["enabled"]:
        return
    logger = get_logger()
    fields["event"] = event
    logger.info(event, extra={"fields": fields})


def enable_logging(
    sample_rate=None,
    url=None,
    batch_size=DEFAULT_BATCH_SIZE,
    auth=None,
    handler=None,
):
    """Wire up the logger. Idempotent — calling twice replaces handlers."""
    if sample_rate is None:
        sample_rate = float(os.environ.get("GPT_PURE_LOG_SAMPLE", DEFAULT_SAMPLE_RATE))
    if url is None:
        url = os.environ.get("GPT_PURE_LOG_URL", DEFAULT_URL)
    if auth is None:
        user = os.environ.get("OO_USER")
        pw = os.environ.get("OO_PASSWORD")
        if user and pw:
            auth = (user, pw)

    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if handler is None:
        handler = OpenObserveHandler(url=url, batch_size=batch_size, auth=auth)
    logger.addHandler(handler)

    _state["sample_rate"] = max(0.0, min(1.0, sample_rate))
    _state["enabled"] = True

    atexit.register(lambda: handler.flush() if hasattr(handler, "flush") else None)
    return handler


def disable_logging():
    logger = get_logger()
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _state["enabled"] = False
    _state["sample_rate"] = 0.0


# Auto-enable from env so `GPT_PURE_LOG=1 python gpt_pure.py` just works.
if os.environ.get("GPT_PURE_LOG") == "1":
    enable_logging()
