from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.demo import router as demo_router
from app.api.generate import router as generate_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Diff Generator API",
        description="Input image from user and generate spot-the-difference outputs.",
        version="0.2.0",
        swagger_ui_parameters={"deepLinking": False},
    )
    app.include_router(generate_router)
    app.include_router(demo_router)

    demo_assets = Path(__file__).resolve().parents[1] / "demo" / "assets"
    if not demo_assets.exists():
        demo_assets = Path(__file__).resolve().parents[2] / "demo" / "assets"
    app.mount("/demo-assets", StaticFiles(directory=str(demo_assets)), name="demo-assets")
    return app
