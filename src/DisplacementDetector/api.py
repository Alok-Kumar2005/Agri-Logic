from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime
import uuid

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from pyproj import Transformer

from src.DisplacementDetector.data_processor import DataProcessor
from src.DisplacementDetector.velocityCalculator import VelocityCalculator
from src.DisplacementDetector.ml_predictor import DisplacementPredictor

router = APIRouter(
    prefix="/analysis/stability",
    tags=["Displacement Detector"]
)

processor: DataProcessor | None = None
predictor: DisplacementPredictor | None = None
calculator = VelocityCalculator()

task_store: Dict[str, Dict] = {}
to_egms = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

def get_processor() -> DataProcessor:
    global processor
    if processor is None:
        processor = DataProcessor()
    return processor


def get_predictor() -> DisplacementPredictor:
    global predictor
    if predictor is None:
        predictor = DisplacementPredictor()
        predictor.load_model()
    return predictor

class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Coordinate(BaseModel):
    latitude: float = Field(..., example=28.6139)
    longitude: float = Field(..., example=77.2090)


class StabilityRequest(BaseModel):
    coordinate: Coordinate


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: Optional[str] = None
    estimated_time: Optional[str] = None


class TimeSeriesPoint(BaseModel):
    date: str
    displacement_mm: float


class StabilityResult(BaseModel):
    task_id: str
    status: TaskStatus
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    mean_velocity_mm_year: Optional[float] = None
    hazard_level: Optional[str] = None
    trend_direction: Optional[str] = None
    acceleration_mm_year2: Optional[float] = None
    temporal_coherence: Optional[float] = None
    time_series: Optional[List[TimeSeriesPoint]] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


def process_stability_task(task_id: str, latitude: float, longitude: float):
    try:
        task_store[task_id]["status"] = TaskStatus.PROCESSING

        proc = get_processor()
        easting, northing = to_egms.transform(longitude, latitude)
        point = proc.find_nearest_point(easting, northing, radius_m=100)
        if not point:
            raise ValueError("No EGMS point found nearby")

        ts: pd.DataFrame = proc.extract_time_series(point)

        analysis = calculator.analyze_point(ts, point)

        history = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "displacement_mm": round(float(row["displacement"]), 2),
            }
            for _, row in ts.iterrows()
        ]

        task_store[task_id].update({
            "status": TaskStatus.COMPLETED,
            "latitude": latitude,
            "longitude": longitude,
            "mean_velocity_mm_year": analysis["mean_velocity_mm_year"],
            "hazard_level": analysis["hazard_level"],
            "trend_direction": analysis["trend_direction"],
            "acceleration_mm_year2": analysis["acceleration_mm_year2"],
            "temporal_coherence": analysis.get("temporal_coherence"),
            "time_series": history,
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        task_store[task_id].update({
            "status": TaskStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })


@router.post("/predict/start", response_model=TaskResponse)
async def start_stability_analysis(
    request: StabilityRequest,
    background_tasks: BackgroundTasks,
):
    task_id = f"stab_{uuid.uuid4().hex[:8]}"

    task_store[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
    }

    background_tasks.add_task(
        process_stability_task,
        task_id,
        request.coordinate.latitude,
        request.coordinate.longitude,
    )

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        estimated_time="20–40s",
        message="Stability analysis started",
    )


@router.get("/predict/results/{task_id}", response_model=StabilityResult)
async def get_stability_results(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return StabilityResult(**task)


@router.get("/predict/tasks")
async def list_tasks():
    return {
        "total_tasks": len(task_store),
        "tasks": [
            {
                "task_id": k,
                "status": v["status"],
                "created_at": v["created_at"],
            }
            for k, v in task_store.items()
        ],
    }
