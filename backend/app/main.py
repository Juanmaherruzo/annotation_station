import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import annotations, classes, export, images, inference, projects
from app.config import settings
from app.core.sam2_backend import SAM2Backend
from app.db.session import init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

sam_engine = SAM2Backend()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    logger.info("Database initialised.")
    logger.info("Loading SAM 2.1 (%s)...", settings.SAM_VARIANT)
    sam_engine.load_model()
    app.state.sam_engine = (
        sam_engine  # accessible from request handlers via request.app.state
    )
    yield
    logger.info("Shutting down — releasing VRAM.")
    sam_engine.unload_model()


app = FastAPI(title="annotation-station API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api")
app.include_router(classes.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(export.router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "sam_loaded": str(sam_engine.is_loaded)}
