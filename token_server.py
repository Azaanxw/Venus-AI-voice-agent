from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from livekit import api

load_dotenv(".env.local")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/token")
async def get_token(
    room: str = "meeting",
    identity: str | None = None,
) -> JSONResponse:
    identity = identity or f"participant-{uuid.uuid4().hex[:6]}"
    token = (
        api.AccessToken(
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return JSONResponse({
        "token": token,
        "url": os.environ["LIVEKIT_URL"],
        "identity": identity,
    })
