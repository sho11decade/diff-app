from fastapi import FastAPI

from app.api.generate import router as generate_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Diff Generator API",
        description="Input image from user and generate spot-the-difference outputs.",
        version="0.2.0",
    )
    app.include_router(generate_router)
    return app
