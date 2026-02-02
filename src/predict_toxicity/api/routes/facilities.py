"""
Industrial Facilities API routes
"""
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from src.predict_toxicity.services.facilities_service import FacilitiesService

router = APIRouter()

class FacilityResponse(BaseModel):
    """Response model for facility data"""
    facility_id: str
    facility_name: str
    city: str
    country: str
    longitude: float
    latitude: float
    sector: str
    sector_code: int
    reporting_year: int
    pollutants: List[dict]

class FacilitySearchResponse(BaseModel):
    """Response model for facility search"""
    total: int
    facilities: List[FacilityResponse]

@router.get("/search", response_model=FacilitySearchResponse)
async def search_facilities(
    country: Optional[str] = Query(None, description="Filter by country"),
    sector: Optional[str] = Query(None, description="Filter by industrial sector"),
    pollutant: Optional[str] = Query(None, description="Filter by pollutant type"),
    year: Optional[int] = Query(None, description="Filter by reporting year"),
    bbox: Optional[str] = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """
    Search industrial facilities with various filters
    
    - **country**: Country name (e.g., "Spain", "Italy")
    - **sector**: Industrial sector name
    - **pollutant**: Pollutant name (e.g., "Lead", "Ammonia")
    - **year**: Reporting year
    - **bbox**: Geographic bounding box
    - **limit**: Maximum number of results
    """
    
    service = FacilitiesService()
    
    filters = {
        "country": country,
        "sector": sector,
        "pollutant": pollutant,
        "year": year,
        "bbox": bbox
    }
    
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    facilities = service.search(filters, limit)
    
    return FacilitySearchResponse(
        total=len(facilities),
        facilities=facilities
    )

@router.get("/{facility_id}", response_model=FacilityResponse)
async def get_facility(facility_id: str):
    """
    Get detailed information about a specific facility
    
    - **facility_id**: Facility identifier
    """
    
    service = FacilitiesService()
    facility = service.get_by_id(facility_id)
    
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")
    
    return facility

@router.get("/{facility_id}/pollutants")
async def get_facility_pollutants(facility_id: str):
    """
    Get list of pollutants released by a facility
    
    - **facility_id**: Facility identifier
    """
    
    service = FacilitiesService()
    pollutants = service.get_pollutants(facility_id)
    
    return {
        "facility_id": facility_id,
        "pollutants": pollutants
    }

@router.get("/{facility_id}/emissions/history")
async def get_emissions_history(
    facility_id: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    """
    Get historical emissions data for a facility
    
    - **facility_id**: Facility identifier
    - **start_year**: Start year for historical data
    - **end_year**: End year for historical data
    """
    
    service = FacilitiesService()
    history = service.get_emissions_history(
        facility_id,
        start_year,
        end_year
    )
    
    return {
        "facility_id": facility_id,
        "emissions_history": history
    }

@router.get("/nearby")
async def get_nearby_facilities(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius_km: float = Query(10.0, ge=0.1, le=100.0, description="Search radius in km"),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Find facilities within a radius of a point
    
    - **lat**: Latitude of center point
    - **lon**: Longitude of center point
    - **radius_km**: Search radius in kilometers
    - **limit**: Maximum number of results
    """
    
    service = FacilitiesService()
    facilities = service.get_nearby(lat, lon, radius_km, limit)
    
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "total": len(facilities),
        "facilities": facilities
    }

@router.get("/statistics/summary")
async def get_statistics_summary(
    country: Optional[str] = None,
    sector: Optional[str] = None,
    year: Optional[int] = None
):
    """
    Get statistical summary of facilities and emissions
    
    - **country**: Filter by country
    - **sector**: Filter by sector
    - **year**: Filter by year
    """
    
    service = FacilitiesService()
    stats = service.get_statistics(country, sector, year)
    
    return stats