import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import AsyncMongoClient

from core.config import settings
from database import db
from llm_serve import LLMRuntime
from qdrant import qdrant
from routers import auth, session, chat
from services.chat_service import ChatService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.client = AsyncMongoClient(settings.MONGODB_URL)
    print("Connected to MongoDB!")
    qdrant_client = qdrant.connect()
    try:
        await qdrant_client.get_collections()
        print(f"Connected to Qdrant at {settings.QDRANT_URL}!")
        app.state.llm_runtime = LLMRuntime.create()
        app.state.chat_service = ChatService(app.state.llm_runtime)
        print(f"TelcoLLM initialized with VLLM at {settings.VLLM_URL}!")
        yield
    finally:
        app.state.chat_service = None
        app.state.llm_runtime = None
        await qdrant.close()
        print("Disconnected from Qdrant!")
        await db.client.close()
        print("Disconnected from MongoDB!")


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(session.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to TelcoLLM Backend API!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
