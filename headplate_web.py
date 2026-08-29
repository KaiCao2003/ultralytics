# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Authenticated web control plane for the headplate YOLO workflow."""

from __future__ import annotations

import os
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from headplate_access import (
    SESSION_MAX_AGE_SECONDS,
    AccessGate,
    login_page,
    normalize_base_path,
    safe_return_url,
    same_origin,
)
from headplate_workflow import (
    configure_project,
    default_config,
    discover_annotation_exports,
    discover_videos,
    launch_worker,
    recover_dead_job,
    resolve_under,
    workflow_root,
    workspace,
)

APP_NAME = "Headplate YOLO"
COOKIE_NAME = "headplate_yolo_session"
CSRF_COOKIE_NAME = "headplate_yolo_csrf"
BASE_PATH = normalize_base_path(os.environ.get("YOLO_BASE_PATH", "/yolo"))
STATIC_DIR = Path(__file__).resolve().parent / "headplate_static"
RUNTIME_DIR = Path(os.environ.get("YOLO_RUNTIME_DIR", Path(__file__).resolve().parent / ".runtime"))


class ProjectRequest(BaseModel):
    """Identify one project below the data root."""

    project: str


class InitializeRequest(ProjectRequest):
    """Initialize a project from videos already on disk."""

    videos: list[str]
    config: dict[str, object]


class ProcessRequest(ProjectRequest):
    """Process one Label Studio export already on disk."""

    annotations: str


def _secure_request(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    return (forwarded or request.url.scheme).casefold() == "https"


def _project(relative: str) -> Path:
    if not relative.strip():
        raise HTTPException(400, "Select a project folder")
    try:
        return resolve_under(workflow_root(), relative)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


def _project_view(project: Path) -> dict[str, object]:
    state = None
    if (workspace(project) / "state.json").exists():
        state = recover_dead_job(project)
    return {
        "project": project.relative_to(workflow_root()).as_posix() or ".",
        "videos": [path.name for path in discover_videos(project)],
        "annotations": [path.name for path in discover_annotation_exports(project)],
        "state": state,
    }


def _gate(request: Request) -> AccessGate:
    gate = getattr(request.app.state, "access_gate", None)
    if gate is None:
        raise HTTPException(503, "Access gate is not initialized")
    return gate


@asynccontextmanager
async def lifespan(application: FastAPI):
    root = workflow_root()
    if not root.is_dir():
        raise RuntimeError(f"YOLO workflow root does not exist: {root}")
    application.state.access_gate = AccessGate.from_environment(RUNTIME_DIR / "access.sqlite3")
    yield


def create_app() -> FastAPI:
    """Create the web application."""
    application = FastAPI(title=APP_NAME, docs_url=None, redoc_url=None, lifespan=lifespan)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.middleware("http")
    async def access_gate(request: Request, call_next):
        route = request.url.path
        public = route == "/health" or route == "/login" or route.startswith("/static/")
        gate = getattr(request.app.state, "access_gate", None)
        token = request.cookies.get(COOKIE_NAME)
        if not public and (gate is None or not gate.validate_session(token)):
            if route.startswith("/api/"):
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
            target = f"{BASE_PATH}{route}"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            return RedirectResponse(f"{BASE_PATH}/login?{urlencode({'next': target})}", status_code=303)
        if (
            not public
            and request.method not in {"GET", "HEAD", "OPTIONS"}
            and (
                not same_origin(
                    request.headers.get("x-forwarded-proto", request.url.scheme),
                    request.headers.get("host"),
                    request.headers.get("origin"),
                    request.headers.get("referer"),
                    request.headers.get("sec-fetch-site"),
                )
                or not gate.validate_csrf(token, request.headers.get("x-csrf-token"))
            )
        ):
            return JSONResponse({"detail": "Request security check failed"}, status_code=403)
        response = await call_next(request)
        for name, value in {
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; base-uri 'none'; connect-src 'self'; font-src 'self'; "
                "form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; object-src 'none'; "
                "script-src 'self'; style-src 'self'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        }.items():
            response.headers.setdefault(name, value)
        return response

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/login", response_class=HTMLResponse)
    async def login_get(request: Request) -> Response:
        return_to = safe_return_url(request.query_params.get("next", ""), BASE_PATH)
        if _gate(request).validate_session(request.cookies.get(COOKIE_NAME)):
            return RedirectResponse(return_to, status_code=303)
        return HTMLResponse(login_page(APP_NAME, f"{BASE_PATH}/login", return_to, False))

    @application.post("/login")
    async def login_post(request: Request) -> Response:
        values = parse_qs((await request.body()).decode("utf-8", errors="replace"), keep_blank_values=True)
        answer = values.get("answer", [""])[-1]
        return_to = safe_return_url(values.get("next", [""])[-1], BASE_PATH)
        gate = _gate(request)
        if not gate.validate_answer(answer):
            return HTMLResponse(login_page(APP_NAME, f"{BASE_PATH}/login", return_to, True), status_code=401)
        token, csrf = gate.issue_session()
        response = RedirectResponse(return_to, status_code=303)
        secure = _secure_request(request)
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=SESSION_MAX_AGE_SECONDS,
            path=BASE_PATH or "/",
            secure=secure,
            httponly=True,
            samesite="strict",
        )
        response.set_cookie(
            CSRF_COOKIE_NAME,
            csrf,
            max_age=SESSION_MAX_AGE_SECONDS,
            path=BASE_PATH or "/",
            secure=secure,
            httponly=False,
            samesite="strict",
        )
        return response

    @application.post("/logout")
    async def logout(request: Request) -> Response:
        _gate(request).revoke(request.cookies.get(COOKIE_NAME))
        response = RedirectResponse(f"{BASE_PATH}/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path=BASE_PATH or "/")
        response.delete_cookie(CSRF_COOKIE_NAME, path=BASE_PATH or "/")
        return response

    @application.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/projects")
    async def projects() -> dict[str, object]:
        root = workflow_root()
        return {
            "root": str(root),
            "projects": sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
            "defaults": default_config(),
        }

    @application.get("/api/project")
    async def project(project: str) -> dict[str, object]:
        return _project_view(_project(project))

    @application.post("/api/initialize")
    async def initialize(payload: InitializeRequest) -> dict[str, object]:
        project = _project(payload.project)
        try:
            configure_project(project, [project / value for value in payload.videos], payload.config)
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        return _project_view(project)

    @application.post("/api/prepare")
    async def prepare(payload: ProjectRequest) -> dict[str, object]:
        project = _project(payload.project)
        try:
            launch_worker(project, "prepare-round1")
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return _project_view(project)

    @application.post("/api/process")
    async def process(payload: ProcessRequest) -> dict[str, object]:
        project = _project(payload.project)
        try:
            launch_worker(project, "process-round", project / payload.annotations)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return _project_view(project)

    @application.get("/api/log")
    async def log(project: str) -> dict[str, object]:
        project_path = _project(project)
        log_path = workspace(project_path) / "workflow.log"
        if not log_path.exists():
            return {"lines": []}
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            return {"lines": list(deque(handle, maxlen=120))}

    return application


app = create_app()
