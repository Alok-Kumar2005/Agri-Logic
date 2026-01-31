from datetime import datetime
from typing import Dict, Optional

from src.chemical_analysis.inference.pipeline import InferencePipeline
from src.chemical_analysis.inference.earth_engine_feature import EarthEngineDataFetcher
from src.chemical_analysis.api.schemas import (
    AnalysisRequest,
    TaskStatus,
    GeometryType,
    Feature,
    FeatureProperties,
    FeatureCollection,
)
from src.logging import logging

task_store: Dict[str, Dict] = {}

_inference_pipeline: Optional[InferencePipeline] = None
_ee_fetcher: Optional[EarthEngineDataFetcher] = None


def get_inference_pipeline() -> InferencePipeline:
    global _inference_pipeline
    if _inference_pipeline is None:
        logging.info("Initializing inference pipeline")
        _inference_pipeline = InferencePipeline(use_finetuned=True)
    return _inference_pipeline


def get_ee_fetcher() -> EarthEngineDataFetcher:
    global _ee_fetcher
    if _ee_fetcher is None:
        logging.info("Initializing Earth Engine fetcher")
        _ee_fetcher = EarthEngineDataFetcher()
    return _ee_fetcher


async def process_analysis_task(task_id: str, request: AnalysisRequest):
    try:
        task_store[task_id]["status"] = TaskStatus.PROCESSING

        fetcher = get_ee_fetcher()

        # Geometry handling
        if request.geometry.type == GeometryType.POLYGON:
            ee_geometry = fetcher.create_geometry_from_coords(
                request.geometry.coordinates[0]
            )
        else:
            lon, lat = request.geometry.coordinates
            ee_geometry = fetcher.create_geometry_from_bbox(
                min_lon=lon - 0.01,
                min_lat=lat - 0.01,
                max_lon=lon + 0.01,
                max_lat=lat + 0.01,
            )

        # Fetch features
        image = fetcher.fetch_all_features(
            geometry=ee_geometry,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        mean_values = fetcher.get_mean_values(
            image=image,
            geometry=ee_geometry,
            scale=100,
        )

        pipeline = get_inference_pipeline()
        preds = pipeline.predict(mean_values)

        # Recommendation logic
        recs = []
        if preds["N"] < 200:
            recs.append("Increase Nitrogen")
        if preds["P"] < 25:
            recs.append("Increase Phosphorus")
        if preds["K"] < 120:
            recs.append("Increase Potassium")

        if preds["pH"] < 5.5:
            recs.append("Apply lime")
        elif preds["pH"] > 8.0:
            recs.append("Apply sulfur")

        recommendation = "; ".join(recs) or "Soil nutrients are balanced"

        feature = Feature(
            geometry=request.geometry,
            properties=FeatureProperties(
                nitrogen=round(preds["N"], 2),
                phosphorus=round(preds["P"], 2),
                potassium=round(preds["K"], 2),
                ph=round(preds["pH"], 2),
                recommendation=recommendation,
            ),
        )

        task_store[task_id].update(
            {
                "status": TaskStatus.COMPLETED,
                "data": FeatureCollection(features=[feature]),
                "completed_at": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}")
        task_store[task_id].update(
            {
                "status": TaskStatus.FAILED,
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat(),
            }
        )
