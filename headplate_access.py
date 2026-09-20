# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared-answer access gate matching the other lab web applications."""

from __future__ import annotations

import hashlib
import hmac
import html
import os
import secrets
import sqlite3
import time
import unicodedata
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit

ANSWER_ENV_NAME = "MOUSELINE_LOGIN_ANSWER"
GENERATION_ENV_NAME = "MOUSELINE_AUTH_GENERATION"
SESSION_MAX_AGE_SECONDS = 30 * 24 * 60 * 60


class AccessGate:
    """Validate the shared answer and persist hashed browser sessions."""

    def __init__(self, answer: str, generation: str, session_db: Path) -> None:
        normalized = self._normalize(answer)
        if not normalized:
            raise ValueError("The access-gate answer must not be empty")
        if not generation.strip():
            raise ValueError("The access-gate generation must not be empty")
        self._answer_digest = hashlib.sha256(normalized.encode()).digest()
        self._generation = generation.strip()
        self.session_db = session_db.resolve()
        self._initialize()

    @classmethod
    def from_environment(cls, session_db: Path) -> AccessGate:
        """Create the gate from the shared lab environment file."""
        answer = os.environ.get(ANSWER_ENV_NAME, "")
        generation = os.environ.get(GENERATION_ENV_NAME, "")
        if not answer.strip() or not generation.strip():
            raise RuntimeError(f"{ANSWER_ENV_NAME} and {GENERATION_ENV_NAME} must be set")
        return cls(answer, generation, session_db)

    @staticmethod
    def _normalize(value: str) -> str:
        return unicodedata.normalize("NFKC", value).strip().casefold()

    @staticmethod
    def _digest(value: str) -> bytes:
        return hashlib.sha256(value.encode()).digest()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.session_db, timeout=5)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        self.session_db.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.session_db.parent, 0o700)
        descriptor = os.open(self.session_db, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        os.chmod(self.session_db, 0o600)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    token_digest BLOB PRIMARY KEY,
                    csrf_digest BLOB NOT NULL,
                    auth_generation TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at)")

    def validate_answer(self, candidate: str) -> bool:
        """Return whether a submitted answer matches without timing leaks."""
        if len(candidate) > 100:
            return False
        digest = hashlib.sha256(self._normalize(candidate).encode()).digest()
        return hmac.compare_digest(self._answer_digest, digest)

    def issue_session(self) -> tuple[str, str]:
        """Create opaque session and CSRF tokens."""
        token, csrf = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (self._digest(token), self._digest(csrf), self._generation, now + SESSION_MAX_AGE_SECONDS),
            )
        return token, csrf

    def validate_session(self, candidate: str | None) -> bool:
        """Return whether a session token is current and unexpired."""
        if not candidate or len(candidate) > 256:
            return False
        now = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT expires_at FROM sessions WHERE token_digest = ? AND auth_generation = ?",
                (self._digest(candidate), self._generation),
            ).fetchone()
            if row and int(row[0]) <= now:
                connection.execute("DELETE FROM sessions WHERE token_digest = ?", (self._digest(candidate),))
                row = None
        return row is not None

    def validate_csrf(self, session: str | None, csrf: str | None) -> bool:
        """Validate a CSRF token bound to an authenticated session."""
        if not session or not csrf or len(session) > 256 or len(csrf) > 256:
            return False
        with self._connect() as connection:
            row = connection.execute(
                "SELECT csrf_digest, expires_at FROM sessions WHERE token_digest = ? AND auth_generation = ?",
                (self._digest(session), self._generation),
            ).fetchone()
        return bool(row and int(row[1]) > int(time.time()) and hmac.compare_digest(bytes(row[0]), self._digest(csrf)))

    def revoke(self, candidate: str | None) -> None:
        """Delete a browser session."""
        if candidate and len(candidate) <= 256:
            with self._connect() as connection:
                connection.execute("DELETE FROM sessions WHERE token_digest = ?", (self._digest(candidate),))


def normalize_base_path(value: str) -> str:
    """Normalize a reverse-proxy base path."""
    value = value.strip()
    if not value or value == "/":
        return ""
    return f"/{value.strip('/')}"


def safe_return_url(value: str, base_path: str) -> str:
    """Allow only application-local login return URLs."""
    base_path = normalize_base_path(base_path)
    fallback = f"{base_path}/"
    if not value or len(value) > 2048 or "\\" in value:
        return fallback
    try:
        parsed = urlsplit(value)
    except ValueError:
        return fallback
    decoded = unquote(parsed.path)
    required = f"{base_path}/" if base_path else "/"
    if parsed.scheme or parsed.netloc or parsed.fragment or not decoded.startswith(required):
        return fallback
    if any(part in {".", ".."} for part in decoded.split("/")):
        return fallback
    return f"{parsed.path}{'?' + parsed.query if parsed.query else ''}"


def _origin_identity(parsed: SplitResult) -> tuple[str, str, int] | None:
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold().rstrip(".")
    if scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    return scheme, host, port or (443 if scheme == "https" else 80)


def same_origin(scheme: str, host: str | None, origin: str | None, referer: str | None, fetch_site: str | None) -> bool:
    """Validate the browser origin for state-changing requests."""
    if (fetch_site or "").casefold() == "same-origin":
        return True
    if not host:
        return False
    expected = _origin_identity(urlsplit(f"{scheme}://{host}"))
    if not expected:
        return False
    candidates = [value for value in (origin, referer) if value]
    return bool(candidates) and all(_origin_identity(urlsplit(value)) == expected for value in candidates)


def login_page(app_name: str, action: str, return_to: str, error: bool) -> str:
    """Return the small login page used across lab apps."""
    error_html = '<div class="error" role="alert">That answer is not correct.</div>' if error else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Lab access · {html.escape(app_name)}</title>
<style>:root{{font-family:ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f4f7fb}}*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center}}main{{width:min(92vw,420px);padding:2rem;border:1px solid #d8e0ec;border-radius:16px;background:#fff;box-shadow:0 14px 40px #15213a1a}}
.brand{{margin:0 0 .35rem;color:#526078;font-size:.82rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}}h1{{margin:0 0 1.5rem;font-size:1.7rem}}label span{{display:block;margin-bottom:.55rem;font-weight:650}}input,button{{width:100%;min-height:44px;border-radius:9px;font:inherit}}input{{padding:.7rem .8rem;border:1px solid #aeb9ca}}button{{margin-top:1rem;border:0;color:#fff;background:#2957c8;font-weight:700;cursor:pointer}}.error{{margin:0 0 1rem;padding:.75rem;border-radius:8px;color:#8b1e1e;background:#fff0f0}}</style></head>
<body><main><p class="brand">{html.escape(app_name)}</p><h1>Lab access</h1>{error_html}
<form method="post" action="{html.escape(action, quote=True)}"><input type="hidden" name="next" value="{html.escape(return_to, quote=True)}">
<label><span>What's the PI's first name?</span><input type="password" name="answer" maxlength="100" autocomplete="off" required autofocus></label>
<button type="submit">Continue</button></form></main></body></html>"""
