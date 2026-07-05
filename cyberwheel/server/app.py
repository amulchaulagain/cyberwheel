"""FastAPI application factory: JSON API + the built single-page app."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from cyberwheel.server import jobs, registry
from cyberwheel.server.paths import STATIC_DIR, ensure_dirs
from cyberwheel.server.validation import ApiError


def create_app() -> FastAPI:
    ensure_dirs()
    # Queued records from a previous server process can never launch (the
    # pending queue lives in process memory); orphan them at boot so their
    # sweeps don't report 'running' forever.
    jobs.orphan_stale_queued()
    app = FastAPI(
        title="Cyberwheel Experimentation",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    from cyberwheel.server.routes import (
        eval_routes,
        network_routes,
        options_routes,
        run_routes,
        sweep_routes,
    )

    app.include_router(options_routes.router)
    app.include_router(run_routes.router)
    app.include_router(eval_routes.router)
    app.include_router(network_routes.router)
    app.include_router(sweep_routes.router)

    @app.get("/api/health")
    def health() -> dict:
        active = sum(1 for r in registry.list_runs() if r.get("status") == "running")
        return {"status": "ok", "active_runs": active}

    @app.get("/{spa_path:path}", include_in_schema=False)
    def spa(spa_path: str):
        if spa_path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": f"/{spa_path}"}},
            )
        candidate = (STATIC_DIR / spa_path).resolve() if spa_path else None
        if (
            candidate
            and candidate.is_file()
            and candidate.is_relative_to(STATIC_DIR.resolve())
        ):
            return FileResponse(candidate)
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index, media_type="text/html")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "no_frontend",
                    "message": "frontend bundle missing — rebuild cyberwheel/server/static",
                }
            },
        )

    return app
