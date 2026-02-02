from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from src.predict_toxicity.services.meteorological_service import MeteorologicalService

router = APIRouter()

class WeatherData(BaseModel):
    """Weather data response model"""
    timestamp: datetime
    location: dict
    temperature_c: float
    wind_speed_ms: float
    wind_direction_deg: float
    pressure_hpa: float
    boundary_layer_height_m: float

class DispersionParameters(BaseModel):
    """Atmospheric dispersion parameters"""
    stability_class: str
    mixing_height_m: float
    wind_u_component: float
    wind_v_component: float
    temperature_c: float
    pressure_hpa: float

@router.get("/current", response_model=WeatherData)
async def get_current_weather(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get current meteorological conditions for a location
    
    - **lat**: Latitude
    - **lon**: Longitude
    """
    
    service = MeteorologicalService()
    weather = service.get_current_weather(lat, lon)
    
    if not weather:
        raise HTTPException(status_code=404, detail="Weather data not available")
    
    return weather

@router.get("/historical")
async def get_historical_weather(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    parameters: Optional[str] = Query(None, description="Comma-separated parameters")
):
    """
    Get historical weather data for a location and time period
    
    - **lat**: Latitude
    - **lon**: Longitude
    - **start_date**: Start date
    - **end_date**: End date
    - **parameters**: Specific parameters to retrieve
    """
    
    service = MeteorologicalService()
    
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    param_list = parameters.split(",") if parameters else None
    
    data = service.get_historical_weather(lat, lon, start, end, param_list)
    
    return {
        "location": {"lat": lat, "lon": lon},
        "period": {"start": start_date, "end": end_date},
        "data": data
    }

@router.get("/dispersion-params", response_model=DispersionParameters)
async def get_dispersion_parameters(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    timestamp: Optional[str] = Query(None, description="ISO timestamp (default: current)")
):
    """
    Get atmospheric dispersion parameters for pollution modeling
    
    - **lat**: Latitude
    - **lon**: Longitude
    - **timestamp**: Specific timestamp (optional)
    """
    
    service = MeteorologicalService()
    
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    else:
        dt = datetime.now()
    
    params = service.get_dispersion_parameters(lat, lon, dt)
    
    return params

@router.get("/wind-field")
async def get_wind_field(
    min_lat: float = Query(..., description="Minimum latitude"),
    min_lon: float = Query(..., description="Minimum longitude"),
    max_lat: float = Query(..., description="Maximum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    timestamp: Optional[str] = Query(None, description="ISO timestamp"),
    resolution: float = Query(0.1, description="Grid resolution in degrees")
):
    """
    Get wind field data for a bounding box (for visualization)
    
    - **min_lat**: Minimum latitude of bounding box
    - **min_lon**: Minimum longitude of bounding box
    - **max_lat**: Maximum latitude of bounding box
    - **max_lon**: Maximum longitude of bounding box
    - **timestamp**: Specific timestamp
    - **resolution**: Grid resolution
    """
    
    service = MeteorologicalService()
    
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    else:
        dt = datetime.now()
    
    wind_field = service.get_wind_field(
        min_lat, min_lon, max_lat, max_lon, dt, resolution
    )
    
    return {
        "bbox": {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon
        },
        "timestamp": dt.isoformat(),
        "resolution": resolution,
        "wind_field": wind_field
    }

@router.get("/forecast")
async def get_forecast(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    hours: int = Query(24, ge=1, le=168, description="Forecast hours ahead")
):
    """
    Get weather forecast for dispersion modeling
    
    - **lat**: Latitude
    - **lon**: Longitude
    - **hours**: Number of hours to forecast
    """
    
    service = MeteorologicalService()
    forecast = service.get_forecast(lat, lon, hours)
    
    return {
        "location": {"lat": lat, "lon": lon},
        "forecast_hours": hours,
        "forecast": forecast
    }