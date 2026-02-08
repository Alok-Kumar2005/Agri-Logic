from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.chemical_analysis.api.routes import router as chemical_analysis_router
from src.predict_toxicity.api.routes.facilities import router as facilities_router
from src.predict_toxicity.api.routes.meteorological import router as meteo_router
from src.predict_toxicity.api.routes.simulation import router as simulation_router
from src.predict_toxicity.api.routes.terrain import router as terrain_router
from src.DisplacementDetector.api import router as displacement_router

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(
    title="Agri-Logic & Toxicity Prediction API",
    version="1.0.0",
    description="Soil Chemical Analysis & Industrial Disaster Simulation APIs",
)

# Chemical Analysis Routes
app.include_router(chemical_analysis_router, prefix="/api")

# Toxicity Prediction Routes
app.include_router(facilities_router, prefix="/api/facilities", tags=["Industrial Facilities"])
app.include_router(meteo_router, prefix="/api/meteorological", tags=["Meteorological Data"])
app.include_router(simulation_router, prefix="/api/simulate", tags=["Disaster Simulation"])
app.include_router(terrain_router, prefix="/api/terrain", tags=["Terrain Analysis"])
app.include_router(displacement_router, prefix="/api/displacement", tags=["Displacement Detector"])

app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

@app.get("/health")
async def health_check():
    return {"status": "ok", "services": ["chemical_analysis", "toxicity_prediction", "displacement_detection"]}

@app.get("/", response_class=FileResponse)
async def root():
    return WEB_DIR / "index.html"

@app.get("/api-info")
async def api_info():
    return {
        "message": "Agri-Logic & Toxicity Prediction API",
        "docs": "/docs",
        "health": "/health"
    }
