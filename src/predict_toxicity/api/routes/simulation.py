"""
Simulation API routes for disaster modeling
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from services.simulation_service import SimulationService
from services.hydrological_service import HydrologicalService
from services.dispersion_service import DispersionService

router = APIRouter()

# Pydantic models for request/response
class CalamityRequest(BaseModel):
    """Request model for calamity simulation"""
    site_id: str = Field(..., description="Industrial facility identifier")
    calamity_type: str = Field(..., description="Type of disaster: flood, earthquake, fire, explosion")
    magnitude: float = Field(..., description="Disaster magnitude")
    unit: str = Field(..., description="Unit of magnitude (e.g., meters_above_base, richter_scale)")
    meteorological_conditions: Optional[Dict] = Field(None, description="Optional weather override")

class SimulationResponse(BaseModel):
    """Response model for simulation initiation"""
    simulation_id: str
    status: str
    engine: str
    estimated_completion_seconds: int
    created_at: datetime

class RiskProfileResponse(BaseModel):
    """Response model for risk profile"""
    simulation_id: str
    critical_radius_km: float
    affected_metrics: Dict
    fallout_geometry: Dict
    health_risks: List[str]
    timestamp: datetime

class SimulationStatus(BaseModel):
    """Response model for simulation status"""
    simulation_id: str
    status: str
    progress_percentage: int
    current_step: str
    error_message: Optional[str] = None

# In-memory storage for demonstration (replace with database in production)
simulations_db = {}

@router.post("/calamity", response_model=SimulationResponse)
async def initiate_calamity_simulation(
    request: CalamityRequest,
    background_tasks: BackgroundTasks
):
    """
    Initiate a disaster simulation for specified industrial site
    
    - **site_id**: Industrial facility identifier (e.g., "ind_site_taloja_44")
    - **calamity_type**: Type of disaster (flood, earthquake, fire, explosion)
    - **magnitude**: Disaster magnitude
    - **unit**: Unit of measurement
    """
    
    # Generate simulation ID
    simulation_id = f"sim_tox_{uuid.uuid4().hex[:6]}"
    
    # Determine simulation engine based on calamity type
    engine_mapping = {
        "flood": "Hydrological_Flow_V1",
        "earthquake": "Seismic_Impact_V1",
        "fire": "Atmospheric_Dispersion_V1",
        "explosion": "Blast_Radius_V1"
    }
    
    engine = engine_mapping.get(request.calamity_type, "Generic_Simulation_V1")
    
    # Create simulation record
    simulation = {
        "simulation_id": simulation_id,
        "site_id": request.site_id,
        "calamity_type": request.calamity_type,
        "magnitude": request.magnitude,
        "unit": request.unit,
        "status": "PROCESSING",
        "engine": engine,
        "created_at": datetime.now(),
        "progress": 0
    }
    
    simulations_db[simulation_id] = simulation
    
    # Add background task to run simulation
    background_tasks.add_task(
        run_simulation,
        simulation_id,
        request.site_id,
        request.calamity_type,
        request.magnitude
    )
    
    return SimulationResponse(
        simulation_id=simulation_id,
        status="PROCESSING",
        engine=engine,
        estimated_completion_seconds=120,
        created_at=simulation.get("created_at")
    )

@router.get("/risk-profile/{sim_id}", response_model=RiskProfileResponse)
async def get_risk_profile(sim_id: str):
    """
    Retrieve comprehensive risk profile for completed simulation
    
    - **sim_id**: Simulation identifier
    """
    
    if sim_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation = simulations_db[sim_id]
    
    if simulation["status"] != "COMPLETED":
        raise HTTPException(
            status_code=400,
            detail=f"Simulation is {simulation['status']}. Wait for completion."
        )
    
    # Retrieve results from simulation service
    service = SimulationService()
    results = service.get_results(sim_id)
    
    return RiskProfileResponse(
        simulation_id=sim_id,
        critical_radius_km=results.get("critical_radius_km", 0.0),
        affected_metrics=results.get("affected_metrics", {}),
        fallout_geometry=results.get("fallout_geometry", {}),
        health_risks=results.get("health_risks", []),
        timestamp=datetime.now()
    )

@router.get("/status/{sim_id}", response_model=SimulationStatus)
async def get_simulation_status(sim_id: str):
    """
    Check current status of running simulation
    
    - **sim_id**: Simulation identifier
    """
    
    if sim_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation = simulations_db[sim_id]
    
    return SimulationStatus(
        simulation_id=sim_id,
        status=simulation["status"],
        progress_percentage=simulation.get("progress", 0),
        current_step=simulation.get("current_step", "Initializing"),
        error_message=simulation.get("error", None)
    )

@router.get("/list")
async def list_simulations(
    status: Optional[str] = None,
    limit: int = 50
):
    """
    List all simulations with optional filtering
    
    - **status**: Filter by status (PROCESSING, COMPLETED, FAILED)
    - **limit**: Maximum number of results
    """
    
    simulations = list(simulations_db.values())
    
    if status:
        simulations = [s for s in simulations if s["status"] == status]
    
    return {
        "total": len(simulations),
        "simulations": simulations[:limit]
    }

async def run_simulation(
    simulation_id: str,
    site_id: str,
    calamity_type: str,
    magnitude: float
):
    """
    Background task to execute simulation
    """
    try:
        simulation = simulations_db[simulation_id]
        
        # Step 1: Load industrial facility data
        simulation["current_step"] = "Loading facility data"
        simulation["progress"] = 10
        
        # Step 2: Load meteorological conditions
        simulation["current_step"] = "Loading meteorological data"
        simulation["progress"] = 25
        
        # Step 3: Load terrain data
        simulation["current_step"] = "Loading terrain data"
        simulation["progress"] = 40
        
        # Step 4: Run hydrological/dispersion model
        simulation["current_step"] = "Running simulation model"
        simulation["progress"] = 60
        
        if calamity_type == "flood":
            hydro_service = HydrologicalService()
            results = hydro_service.simulate_flood(site_id, magnitude)
        else:
            dispersion_service = DispersionService()
            results = dispersion_service.simulate_dispersion(site_id, calamity_type, magnitude)
        
        # Step 5: Calculate impact metrics
        simulation["current_step"] = "Calculating impact metrics"
        simulation["progress"] = 80
        
        # Step 6: Generate risk profile
        simulation["current_step"] = "Generating risk profile"
        simulation["progress"] = 95
        
        # Store results
        simulation["results"] = results
        simulation["status"] = "COMPLETED"
        simulation["progress"] = 100
        simulation["current_step"] = "Completed"
        
    except Exception as e:
        simulation["status"] = "FAILED"
        simulation["error"] = str(e)
        simulation["current_step"] = "Failed"