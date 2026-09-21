from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import router as api_router
from backend.services.db import init_db

# get project root and frontend path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize database when the app starts
    init_db()
    yield


app = FastAPI(
    title="Intelligent Hybrid Aerial-Grounded Warehouse Automation System",
    description="RL-driven warehouse resource allocation simulation (robotic arms, "
                "ground robots and aerial drones) with a live web dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

# allow frontend requests to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# connect API routes
app.include_router(api_router)

# serve frontend static files
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")


@app.get("/")
def serve_index():
    # return the main frontend page
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    # check if the backend is running
    return {"status": "ok"}