import os
import warnings
from dotenv import load_dotenv

# Suppress Pydantic deprecation warnings coming from dependencies like Cohere
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*The `__fields__` attribute is deprecated.*")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import db
from routers import chat, emergency

load_dotenv()

app = FastAPI(
    title="Chatbot API",
    description="Chatbot API for AI-powered PDF analysis and Q&A",
    version="1.0.0"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://poetic-begonia-0234b0.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router)
app.include_router(emergency.router)

@app.on_event("startup")
async def startup():
    print("✅ FASTAPI STARTUP: Connecting Prisma...")
    await db.connect()
    print("✅ FASTAPI STARTUP: Prisma Connected!")

@app.on_event("shutdown")
async def shutdown():
    print("🔻 FASTAPI SHUTDOWN: Disconnecting Prisma...")
    await db.disconnect()

@app.get("/")
async def root():
    return {"message": "Hello World"}
