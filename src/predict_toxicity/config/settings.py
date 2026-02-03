from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "Predictive Toxicity & Fallout Simulation"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    SECRET_KEY: str = "your-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/toxicity_db"
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    
    # Industrial Facilities Data
    INDUSTRIAL_AIR_RELEASES_PATH: Path = RAW_DATA_DIR / "industrial" / "air_releases.csv"
    INDUSTRIAL_WATER_RELEASES_PATH: Path = RAW_DATA_DIR / "industrial" / "water_releases.csv"
    
    # Meteorological Data
    ERA5_DATA_PATH: Path = RAW_DATA_DIR / "meteorological" / "data_stream.nc"
    
    # Terrain Data
    DEM_PATH: Path = RAW_DATA_DIR / "terrain" / "elevation.tif"
    SLOPE_PATH: Path = PROCESSED_DATA_DIR / "terrain" / "slope.tif"
    ROUGHNESS_PATH: Path = PROCESSED_DATA_DIR / "terrain" / "roughness.tif"
    FLOW_DIRECTION_PATH: Path = PROCESSED_DATA_DIR / "terrain" / "flow_direction.tif"
    FLOW_ACCUMULATION_PATH: Path = PROCESSED_DATA_DIR / "terrain" / "flow_accumulation.tif"
    
    # Land Use Data
    CORINE_DATA_PATH: Path = RAW_DATA_DIR / "landuse" / "corine"
    
    # Simulation Parameters
    DEFAULT_SIMULATION_RESOLUTION: int = 100  # meters
    MAX_SIMULATION_RADIUS_KM: float = 50.0
    TOXICITY_THRESHOLD_PPM: float = 10.0
    
    # WHO/CPCB Safety Limits (example values - adjust based on pollutant)
    SAFETY_LIMITS: dict = {
        "Lead": 0.01,  # ppm
        "Chromium VI": 0.05,
        "Copper": 1.3,
        "Ammonia": 25.0
    }
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure data directories exist
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)