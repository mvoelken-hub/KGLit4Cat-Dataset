from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import logger, setup_logging

from app.neo4j.driver import neo4j_driver
from app.ollama.client import ollama_client
from app.core.task_registry import task_registry

from app.core.setup import start_setup

# Import API routers
from app.api.v1.system import router as system_router


V1_PREFIX = "/api/v1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup actions can be added here ---

    setup_logging(runtime_dir=settings.runtime_dir)

    # Start setup tasks
    await start_setup(
        settings=settings,
        logger=logger,
        ollama_client=ollama_client,
        neo4j_driver=neo4j_driver,
        task_registry=task_registry
    )

    # ---------------------------------------------------------------------------

    yield
    # --- Shutdown actions can be added here ---

    # Cancel all remaining tasks in the task registry
    await task_registry.cancel_all_tasks()
    
    # Close Neo4j driver connection
    await neo4j_driver.close()

    # Close Ollama client connection
    await ollama_client.close()

# ---------------------------------------------------------------------------

# Create FastAPI app with lifespan management for startup and shutdown actions
fastapi_app = FastAPI(title="voc4cat-extraction API", lifespan=lifespan)

# Configure CORS middleware to allow requests from the frontend application
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_base_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a root endpoint that redirects to the API documentation
@fastapi_app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


# Register routes
fastapi_app.include_router(system_router, prefix=V1_PREFIX)
