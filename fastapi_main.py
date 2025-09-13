
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
import os

from routes.policy_routes import policy_router
from routes.policy_assistant_routes import policy_assistant_router
from database.database import init_db

app = FastAPI(
    title="Organization Policy Management API",
    description="A comprehensive API for managing organizational policies with AI-powered assistance",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
@app.on_event("startup")
def startup_event():
    init_db()

# Include routers
app.include_router(policy_router, prefix="/api/v1/policies", tags=["policies"])
app.include_router(policy_assistant_router, prefix="/api/v1/assistant", tags=["policy-assistant"])

# ✅ Correct path to frontend folder
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
async def serve_vue_frontend():
    return FileResponse(os.path.join(frontend_path, "vue_frontend.html"))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
