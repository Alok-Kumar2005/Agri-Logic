from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GeometryType(str, Enum):
    POLYGON = "Polygon"
    POINT = "Point"


class Geometry(BaseModel):
    type: GeometryType
    coordinates: list

class AnalysisRequest(BaseModel):
    aoi_name: str = Field(..., description="Area of Interest name")
    geometry: Geometry
    crop_type: Optional[str] = "Unknown"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class FeatureProperties(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    recommendation: str


class Feature(BaseModel):
    type: str = "Feature"
    geometry: Geometry
    properties: FeatureProperties


class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[Feature]

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    estimated_time: Optional[str] = None
    message: Optional[str] = None


class AnalysisResult(BaseModel):
    task_id: str
    status: TaskStatus
    data: Optional[FeatureCollection] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
