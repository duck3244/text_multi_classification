"""FastAPI entrypoint for the Korean UnSmile classifier MVP.

Run from project root (or any cwd):
    uvicorn backend.api.app:app --reload --port 8000

Or from backend/:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from api.inference import InferenceEngine  # noqa: E402
from api.schemas import (  # noqa: E402
    HealthResponse,
    LabelMeta,
    LabelsResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델/토크나이저 1회 로드 후 메모리 상주
    engine = InferenceEngine()
    engine.load()
    app.state.engine = engine
    yield
    # 종료 시 정리할 리소스 없음 (PyTorch 모델은 프로세스 종료 시 자동 해제)


app = FastAPI(
    title="Korean UnSmile Classifier API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: InferenceEngine = app.state.engine
    return HealthResponse(
        status="ok",
        model_loaded=engine.loaded,
        device=str(engine.device),
    )


@app.get("/api/labels", response_model=LabelsResponse)
async def labels() -> LabelsResponse:
    engine: InferenceEngine = app.state.engine
    return LabelsResponse(
        labels=[
            LabelMeta(name=name, description=engine.label_descriptions.get(name, ""))
            for name in engine.label_columns
        ]
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    engine: InferenceEngine = app.state.engine
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    # PyTorch 추론은 동기 — 이벤트 루프 블로킹 방지를 위해 스레드풀로
    result = await run_in_threadpool(engine.predict, req.text, req.threshold)
    return PredictResponse(**result)
