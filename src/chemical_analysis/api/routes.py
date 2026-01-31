import uuid
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from src.chemical_analysis.api.schemas import (
    AnalysisRequest,
    TaskResponse,
    TaskStatus,
    AnalysisResult,
)
from src.chemical_analysis.api.functions import (
    task_store,
    process_analysis_task,
)
from src.logging import logging

router = APIRouter(tags=["Chemical Analysis"])


@router.post("/analysis/agri/start", response_model=TaskResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks ):
    task_id = f"agri_{uuid.uuid4().hex[:8]}"

    task_store[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "aoi_name": request.aoi_name,
        "created_at": datetime.utcnow().isoformat(),
        "data": None,
        "error": None,
        "completed_at": None,
    }

    background_tasks.add_task(process_analysis_task, task_id, request)

    logging.info(f"Analysis task created: {task_id}")

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        estimated_time="45s",
        message="Analysis started",
    )

@router.get("/analysis/agri/results/{task_id}", response_model=AnalysisResult)
async def get_analysis_results(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return AnalysisResult(**task)

@router.get("/analysis/agri/tasks")
async def list_tasks():
    return {
        "total_tasks": len(task_store),
        "tasks": [
            {
                "task_id": k,
                "status": v["status"],
                "aoi_name": v["aoi_name"],
                "created_at": v["created_at"],
            }
            for k, v in task_store.items()
        ],
    }
