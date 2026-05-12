from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.admin_api import router as admin_router
from server.auth_api import router as auth_router
from server.config import Settings, load_settings
from server.db import bootstrap_database
from server.jobs.api import router as jobs_router
from server.jobs.artifacts import router as artifacts_router
from server.jobs.events import router as events_router
from server.projects.api import router as projects_router
from server.projects.editor_api import router as editor_router
from server.projects.export_api import router as export_router
from server.projects.setup_api import router as setup_router
from server.workers.api import router as workers_router
from server.workers.lease import fail_lost_jobs


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    bootstrap_database(settings)
    app = FastAPI(title="TraduzAi Web", version="0.1.0")
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(jobs_router)
    app.include_router(events_router)
    app.include_router(artifacts_router)
    app.include_router(setup_router)
    app.include_router(projects_router)
    app.include_router(editor_router)
    app.include_router(export_router)
    app.include_router(workers_router)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.on_event("startup")
    async def start_janitor():
        async def _loop():
            while True:
                fail_lost_jobs(settings)
                await asyncio.sleep(30)

        app.state.janitor_task = asyncio.create_task(_loop())

    site_dist = Path(__file__).resolve().parents[1] / "site" / "dist"
    if site_dist.exists():
        app.mount("/", SPAStaticFiles(directory=site_dist, html=True), name="site")
    return app


app = create_app()
