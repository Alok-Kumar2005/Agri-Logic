from fastapi import FastAPI
from src.chemical_analysis.api.routes import router as chemical_analysis_router

app = FastAPI(
    title="Agri-Logic API",
    version="1.0.0",
    description="Soil & Chemical Analysis APIs for Agriculture",
)

app.include_router(chemical_analysis_router, prefix= "/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}