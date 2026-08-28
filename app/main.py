import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import init_db
from app.api.web_routes import router as web_router
from app.api.call_routes import router as call_router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables on startup
    print("🚀 Initializing Vectra-Orbit database tables...")
    await init_db()
    print("✅ Database ready.")
    yield
    print("👋 Shutting down Vectra-Orbit application.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    debug=settings.DEBUG
)

# Mount static files for web console
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Include Routers
app.include_router(web_router)
app.include_router(call_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
