from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.predict_toxicity.services.terrain_service import TerrainService

router = APIRouter()

class ElevationResponse(BaseModel):
    """Elevation data response"""
    location: dict
    elevation_m: float
    source: str

class TerrainProfile(BaseModel):
    """Terrain profile response"""
    points: List[dict]
    total_distance_m: float
    elevation_gain_m: float
    elevation_loss_m: float

@router.get("/elevation", response_model=ElevationResponse)
async def get_elevation(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get elevation at a specific point
    
    - **lat**: Latitude
    - **lon**: Longitude
    """
    
    service = TerrainService()
    elevation = service.get_elevation(lat, lon)
    
    if elevation is None:
        raise HTTPException(status_code=404, detail="Elevation data not available")
    
    return ElevationResponse(
        location={"lat": lat, "lon": lon},
        elevation_m=elevation,
        source="Copernicus DEM GLO-30"
    )

@router.get("/slope")
async def get_slope(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get terrain slope at a specific point
    
    - **lat**: Latitude
    - **lon**: Longitude
    """
    
    service = TerrainService()
    slope = service.get_slope(lat, lon)
    
    if slope is None:
        raise HTTPException(status_code=404, detail="Slope data not available")
    
    return {
        "location": {"lat": lat, "lon": lon},
        "slope_degrees": slope,
        "slope_percent": slope * 1.745
    }

@router.get("/roughness")
async def get_roughness(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get terrain roughness at a specific point
    
    - **lat**: Latitude
    - **lon**: Longitude
    """
    
    service = TerrainService()
    roughness = service.get_roughness(lat, lon)
    
    if roughness is None:
        raise HTTPException(status_code=404, detail="Roughness data not available")
    
    return {
        "location": {"lat": lat, "lon": lon},
        "roughness_m": roughness
    }

@router.get("/flow-direction")
async def get_flow_direction(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get hydrological flow direction at a point
    
    - **lat**: Latitude
    - **lon**: Longitude
    
    Returns D8 flow direction (1-8):
    1=E, 2=SE, 3=S, 4=SW, 5=W, 6=NW, 7=N, 8=NE
    """
    
    service = TerrainService()
    direction = service.get_flow_direction(lat, lon)
    
    if direction is None:
        raise HTTPException(status_code=404, detail="Flow direction data not available")
    
    direction_names = {
        0: "No flow",
        1: "East",
        2: "Southeast",
        3: "South",
        4: "Southwest",
        5: "West",
        6: "Northwest",
        7: "North",
        8: "Northeast"
    }
    
    return {
        "location": {"lat": lat, "lon": lon},
        "flow_direction_code": int(direction),
        "flow_direction_name": direction_names.get(int(direction), "Unknown")
    }

@router.get("/flow-accumulation")
async def get_flow_accumulation(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get flow accumulation (upstream contributing area)
    
    - **lat**: Latitude
    - **lon**: Longitude
    """
    
    service = TerrainService()
    accumulation = service.get_flow_accumulation(lat, lon)
    
    if accumulation is None:
        raise HTTPException(status_code=404, detail="Flow accumulation data not available")
    
    return {
        "location": {"lat": lat, "lon": lon},
        "flow_accumulation": float(accumulation),
        "description": "Number of upstream cells contributing flow"
    }

@router.get("/profile", response_model=TerrainProfile)
async def get_terrain_profile(
    start_lat: float = Query(..., description="Start latitude"),
    start_lon: float = Query(..., description="Start longitude"),
    end_lat: float = Query(..., description="End latitude"),
    end_lon: float = Query(..., description="End longitude"),
    num_points: int = Query(100, ge=10, le=1000, description="Number of sample points")
):
    """
    Get elevation profile between two points
    
    - **start_lat**: Start point latitude
    - **start_lon**: Start point longitude
    - **end_lat**: End point latitude
    - **end_lon**: End point longitude
    - **num_points**: Number of points to sample
    """
    
    service = TerrainService()
    profile = service.get_terrain_profile(
        start_lat, start_lon, end_lat, end_lon, num_points
    )
    
    return profile

@router.get("/watershed")
async def get_watershed(
    lat: float = Query(..., description="Outlet point latitude"),
    lon: float = Query(..., description="Outlet point longitude"),
    threshold: Optional[float] = Query(1000, description="Flow accumulation threshold")
):
    """
    Delineate watershed for a given outlet point
    
    - **lat**: Outlet point latitude
    - **lon**: Outlet point longitude
    - **threshold**: Minimum flow accumulation for stream definition
    """
    
    service = TerrainService()
    watershed = service.delineate_watershed(lat, lon, threshold)
    
    return {
        "outlet": {"lat": lat, "lon": lon},
        "watershed": watershed
    }

@router.get("/aspect")
async def get_aspect(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """
    Get terrain aspect (direction slope faces) at a point
    
    - **lat**: Latitude
    - **lon**: Longitude
    """
    
    service = TerrainService()
    aspect = service.get_aspect(lat, lon)
    
    if aspect is None:
        raise HTTPException(status_code=404, detail="Aspect data not available")
    
    # Convert aspect to cardinal direction
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    index = int((aspect + 22.5) / 45) % 8
    
    return {
        "location": {"lat": lat, "lon": lon},
        "aspect_degrees": aspect,
        "aspect_direction": directions[index]
    }