# 아키텍처 문서 (Architecture)

본 문서는 **Korean UnSmile Multi-label Classification** 프로젝트의 전체 시스템 구조, 모듈 책임, 데이터 흐름, 배포 토폴로지를 설명합니다.

---

## 1. 개요

본 프로젝트는 한국어 텍스트에서 혐오 표현을 탐지·분류하는 **다중 레이블 분류 (Multi-label Classification)** 시스템이며, 크게 두 영역으로 나뉜다.

| 영역 | 역할 | 주요 기술 |
|---|---|---|
| **Backend** | 학습 파이프라인 + 추론 API 서비스 | Python 3.10, PyTorch, HuggingFace Transformers, FastAPI, Uvicorn |
| **Frontend** | 사용자가 텍스트를 입력하여 분류 결과를 확인하는 SPA | Vue 3, TypeScript, Vite, Tailwind CSS v3.4 |

학습된 모델 체크포인트(`best_model.pth`)가 두 영역을 잇는 매개체이며, **학습 단(Training Plane)** 과 **서빙 단(Serving Plane)** 은 의존성/실행 주체가 분리되어 있다.

---

## 2. 레이어 구조

```
┌──────────────────────────────────────────────────────────────────┐
│                       Presentation Layer                         │
│  Vue 3 SPA (App.vue) — 입력 폼 / 임계값 슬라이더 / 결과 / 히스토리   │
└──────────────────────────────────────────────────────────────────┘
                        │ HTTP (fetch /api)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      API / Service Layer                         │
│  FastAPI (api/app.py)                                            │
│   ├ GET  /api/health   → 모델 상태 / device                      │
│   ├ GET  /api/labels   → 10개 레이블 메타                        │
│   └ POST /api/predict  → 텍스트 + threshold 추론                 │
│  InferenceEngine (api/inference.py) — 싱글톤, 모델 1회 로드      │
└──────────────────────────────────────────────────────────────────┘
                        │ tensor I/O
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Model Layer                              │
│  MultiLabelElectraClassifier (model.py)                          │
│   - KoELECTRA encoder + Dropout + Linear(num_labels=10)          │
│   - BCEWithLogitsLoss / WeightedBCE / FocalLoss                  │
└──────────────────────────────────────────────────────────────────┘
                        ▲
                        │ 학습 산출물 (best_model.pth)
                        │
┌──────────────────────────────────────────────────────────────────┐
│                     Training / Eval Pipeline                     │
│  main.py (CLI) → KoreanUnsmileTrainer / ModelEvaluator           │
│   ├ data_processor.py (HF Dataset → CSV + label_info.json)       │
│   ├ trainer.py        (DataLoader, optimizer, scheduler, loop)   │
│   ├ evaluator.py      (metrics, threshold sweep, 시각화)         │
│   ├ utils.py          (MetricsCalculator, EarlyStopping, viz)    │
│   └ config.py         (Experiment/Model/Training/Data/Logging)   │
└──────────────────────────────────────────────────────────────────┘
                        ▲
                        │ HuggingFace Datasets API
                        │
┌──────────────────────────────────────────────────────────────────┐
│                        Data Source Layer                         │
│  smilegate-ai/kor_unsmile (HuggingFace Hub) → korean_unsmile_csv │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 디렉토리 매핑

```
text_multi_classification/
├── backend/
│   ├── main.py                    # 학습 파이프라인 CLI 엔트리포인트
│   ├── config.py                  # ExperimentConfig + ConfigManager (5종 프리셋)
│   ├── data_processor.py          # KoreanUnsmileProcessor — HF 다운로드/전처리
│   ├── model.py                   # MultiLabelElectraClassifier + Dataset + Loss
│   ├── trainer.py                 # KoreanUnsmileTrainer — 학습 루프
│   ├── evaluator.py               # ModelEvaluator — 평가/리포트/시각화
│   ├── utils.py                   # MetricsCalculator/EarlyStopping/Viz/seed/logging
│   ├── api/
│   │   ├── app.py                 # FastAPI + lifespan + CORS
│   │   ├── inference.py           # InferenceEngine (경량 추론)
│   │   └── schemas.py             # Pydantic I/O 스키마
│   ├── korean_unsmile_csv/        # 처리된 CSV + label_info.json + class_weights.json
│   ├── output/<run_name>/         # 학습 산출물 (.gitignore)
│   ├── requirements.txt           # 학습/평가 풀 의존성
│   └── requirements-api.txt       # API 경량 의존성 (matplotlib/sklearn 제외)
└── frontend/
    ├── vite.config.ts             # /api → http://127.0.0.1:8000 프록시
    ├── index.html, main.ts, App.vue
    └── src/lib/
        ├── api.ts                 # fetch 래퍼 + TS 타입
        └── storage.ts             # localStorage 기반 히스토리 (max 10)
```

`backend/config.py:resolve_path()` 는 모든 상대 경로를 `backend/` 기준으로 해석하므로 **어떤 cwd에서 실행하든 동일하게 동작** 한다. `main.py` 와 `api/app.py` 는 `sys.path` 에 `backend/` 를 동적으로 추가하여 형제 모듈 import 를 보장한다.

---

## 4. 핵심 컴포넌트 책임 (SRP)

### 4.1 Backend — Training Plane

| 컴포넌트 | 클래스/모듈 | 책임 |
|---|---|---|
| **설정 관리** | `ExperimentConfig`, `ConfigManager` | 5종 프리셋(default/debug/large_batch/high_lr/conservative), JSON 저장/로드, CLI 인자 머지 |
| **데이터 처리** | `KoreanUnsmileProcessor` | HF dataset 다운로드, 10개 레이블 컬럼 전개, 클래스 가중치 계산, 통계 산출, CSV 저장 |
| **데이터셋** | `KoreanUnsmileDataset (Dataset)` | 토크나이즈 + tensor 변환 |
| **모델** | `MultiLabelElectraClassifier (nn.Module)` | KoELECTRA encoder + dropout + linear head, `predict()` 헬퍼, freeze/unfreeze |
| **손실** | `WeightedBCEWithLogitsLoss`, `FocalLoss` | 클래스 불균형 대응 |
| **학습 루프** | `KoreanUnsmileTrainer` | AdamW + linear/cosine warmup, gradient accumulation, gradient clipping, eval-by-steps/epoch, best-model 저장, early stopping |
| **평가** | `ModelEvaluator` | 임계값 민감도 분석, per-label 메트릭, 분류 리포트, 어려운 샘플 분석, 시각화 |
| **유틸리티** | `MetricsCalculator`, `EarlyStopping`, `TrainingHistory`, `VisualizationUtils` | macro/micro 메트릭, Hamming Loss, AUC, 혼동 행렬, 학습 곡선 |

### 4.2 Backend — Serving Plane

| 컴포넌트 | 책임 | 비고 |
|---|---|---|
| `FastAPI app` (`api/app.py`) | HTTP 엔드포인트 노출, CORS (origin: `localhost:5173`), lifespan 으로 엔진 1회 로드 | 동기 추론은 `run_in_threadpool` 로 이벤트 루프 보호 |
| `InferenceEngine` (`api/inference.py`) | 모델 + 토크나이저 + label_info 1회 로드 후 메모리 상주, `@torch.inference_mode()` 로 grad 비활성화 | `MODEL_CHECKPOINT_PATH`, `LABEL_INFO_PATH` 환경변수로 override 가능 |
| `schemas.py` | Pydantic `PredictRequest/Response`, `LabelMeta`, `HealthResponse` | text 길이 1~4000, threshold 0~1 검증 |

> Serving Plane 은 `matplotlib`/`seaborn`/`sklearn` 의존을 일부러 끊어두어, `requirements-api.txt` 만 설치해도 API 서버가 뜬다.

### 4.3 Frontend

| 컴포넌트 | 책임 |
|---|---|
| `App.vue` | 단일 화면 SPA — 텍스트 입력, threshold 슬라이더(0~1, step 0.05), 분류 결과(확률 정렬 + 막대), 최근 분류 10건 히스토리 |
| `lib/api.ts` | `/api/health`, `/api/labels`, `/api/predict` 호출 + TS 타입 정의 |
| `lib/storage.ts` | `localStorage` 기반 히스토리 (key: `unsmile.history.v1`, max 10) |
| `vite.config.ts` | dev 서버에서 `/api` → `http://127.0.0.1:8000` 프록시 — CORS 우회 |

---

## 5. 데이터 흐름

### 5.1 학습 파이프라인 (`python backend/main.py pipeline`)

```
HuggingFace Hub (smilegate-ai/kor_unsmile)
        │ load_dataset()
        ▼
KoreanUnsmileProcessor.process()
   ├─ process_split(train/valid)        # 10개 레이블 컬럼 전개
   ├─ calculate_class_weights()         # neg/pos 비율
   ├─ calculate_statistics()
   └─ save_*  ───────────────────────► korean_unsmile_csv/
                                          ├─ korean_unsmile_train.csv
                                          ├─ korean_unsmile_valid.csv
                                          ├─ label_info.json
                                          ├─ class_weights.json
                                          └─ dataset_statistics.json
        │
        ▼
KoreanUnsmileTrainer.train()
   ├─ load_data()                       # CSV → DataLoader
   ├─ create_model_and_optimizer()      # KoELECTRA + AdamW + scheduler
   ├─ for epoch:
   │     train_epoch()  ── tqdm + grad accumulation + grad clip
   │     evaluate()     ── MetricsCalculator
   │     check_early_stopping() ─ best_model.pth 저장
   └─ create_visualizations()
        │
        ▼
output/<run_name>/
   ├─ checkpoints/best_model.pth        # ←── 서빙단에서 로드
   ├─ experiment_config.json
   ├─ final_metrics.json
   ├─ training_history.json
   └─ plots/*.png
        │
        ▼
ModelEvaluator.create_evaluation_report()
   └─ evaluation/evaluation_report.json + plots/*
```

### 5.2 추론 파이프라인 (서빙)

```
사용자 (브라우저)
    │ 텍스트 입력 + threshold
    ▼
App.vue          ── POST /api/predict { text, threshold }
    │ (Vite proxy: /api → :8000)
    ▼
FastAPI /api/predict
    │ run_in_threadpool(engine.predict, ...)
    ▼
InferenceEngine.predict()
    ├─ tokenizer(text, max_length=512)
    ├─ model.forward(input_ids, attention_mask)
    ├─ sigmoid(logits)  → 10개 확률
    └─ {labels: [{name, probability, predicted}], top}
    │
    ▼
PredictResponse (JSON)
    │
    ▼
App.vue
   ├─ 확률 내림차순 정렬 후 막대 차트로 표시
   └─ pushHistory() → localStorage (최대 10건)
```

---

## 6. 모델 아키텍처 상세

```
Input (text)
    │
    ▼  ElectraTokenizer (vocab_size=35000, max_length=512)
[input_ids, attention_mask]
    │
    ▼  KoELECTRA encoder (monologg/koelectra-base-v3-discriminator)
       12 layers · hidden=768 · 12 attention heads
last_hidden_state [B, L, 768]
    │
    ▼  [CLS] 토큰 추출  →  [B, 768]
pooled_output
    │
    ▼  Dropout(p=0.1)
    │
    ▼  Linear(768 → 10)
logits  [B, 10]
    │
    ▼  학습: BCEWithLogitsLoss / WeightedBCE / Focal
    ▼  추론: sigmoid → threshold 비교
[label_i: probability, predicted]  × 10
```

- **다중 레이블 헤드**: softmax 가 아닌 **sigmoid** + per-class threshold. 한 문장이 동시에 여러 카테고리로 분류 가능 (예: `여성/가족` + `악플/욕설`).
- **클래스 불균형**: `class_weights.json` (`neg/pos` 비율) 을 `BCEWithLogitsLoss(pos_weight=...)` 에 주입.
- **재현성**: `set_seed()` 가 random/numpy/torch/cudnn 을 모두 결정론적으로 설정.

---

## 7. 설정 시스템

`config.py` 는 4개의 `@dataclass` 를 합성해 하나의 `ExperimentConfig` 를 구성한다.

| dataclass | 주요 필드 |
|---|---|
| `ModelConfig` | `model_name`, `num_labels`, `max_length`, `dropout_rate`, `loss_type`, `focal_*` |
| `TrainingConfig` | `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `warmup_ratio`, `scheduler_type`, `eval_steps/strategy`, `early_stopping_*`, `gradient_accumulation_steps`, `fp16` |
| `DataConfig` | `data_dir`, `train_file`, `valid_file`, `text_column`, `label_columns`, `max_samples` |
| `LoggingConfig` | `output_dir`, `run_name`, `use_wandb/tensorboard`, `save_predictions` |

우선순위: **CLI 인자 > config 파일 > preset > dataclass 기본값**. 모든 경로는 `resolve_path()` 로 `BASE_DIR(=backend/)` 기준 절대 경로화된다.

---

## 8. 배포 토폴로지 (MVP)

```
┌──────────────────────┐         ┌───────────────────────┐
│ Browser              │         │ Vite dev server :5173 │
│  Vue SPA             │ ◀──────▶│  (HMR, /api proxy)    │
└──────────────────────┘         └──────────┬────────────┘
                                            │ /api/*
                                            ▼
                                  ┌───────────────────────┐
                                  │ Uvicorn :8000         │
                                  │  FastAPI app          │
                                  │  └ InferenceEngine    │
                                  │     ├ KoELECTRA model │
                                  │     └ Tokenizer       │
                                  └──────────┬────────────┘
                                             │ load_state_dict
                                             ▼
                                  backend/output/<run>/checkpoints/
                                             best_model.pth
```

- **개발 모드**: `uvicorn backend.api.app:app --reload --port 8000` + `cd frontend && npm run dev`.
- **체크포인트 경로 override**: `MODEL_CHECKPOINT_PATH=/abs/path/best_model.pth uvicorn ...`.
- **CORS**: 현재 `localhost:5173`, `127.0.0.1:5173` 만 허용 (`app.py:50`). 프로덕션 배포 시 origin 확장 필요.
- **GPU 활용**: CUDA 가용 시 자동으로 `cuda` 디바이스 선택, 아니면 CPU.

---

## 9. 확장 / 변경 포인트

| 변경 의도 | 손대야 할 파일 |
|---|---|
| 다른 사전학습 모델로 교체 (예: KoBERT) | `ModelConfig.model_name`, `ElectraModel` → 해당 모델 클래스로 교체 |
| 새 손실 함수 추가 | `model.py` 에 클래스 추가 + `trainer.py:_build_criterion()` 분기 |
| 새 API 엔드포인트 (배치 예측 등) | `api/schemas.py` + `api/app.py` + `InferenceEngine` 메서드 |
| 새 프리셋 추가 | `config.py:ConfigManager._create_*_config()` + `presets` 딕셔너리 |
| 새 시각화 / 메트릭 | `utils.py:VisualizationUtils` / `MetricsCalculator` |
| 프론트 신규 화면 | `frontend/src/` 에 컴포넌트 추가, `api.ts` 에 메서드 추가 |

---

## 10. 알려진 설계 결정

1. **Training Plane 과 Serving Plane 의존성 분리** — `requirements-api.txt` 는 matplotlib/seaborn/sklearn 을 제외. API 서버의 부팅 시간과 컨테이너 크기를 작게 유지.
2. **`InferenceEngine` 싱글톤** — 매 요청마다 모델을 다시 로드하지 않도록 FastAPI `lifespan` 에서 1회 로드 후 `app.state.engine` 에 보관.
3. **동기 추론 + threadpool** — PyTorch 추론은 동기지만 `run_in_threadpool` 로 이벤트 루프를 막지 않도록 우회.
4. **상대 경로 정규화** — `resolve_path()` + `BASE_DIR` 패턴으로 cwd 의존성 제거.
5. **시각화 라벨 영문화** — `utils.py:LABEL_EN_MAP` 으로 matplotlib 한글 폰트 의존 회피. CSV/JSON 의 원본 한글 레이블은 그대로 유지.
6. **Frontend 상태 영속화는 localStorage 만** — 백엔드에 사용자/세션 개념 없음. MVP 단계에서는 의도적으로 단순화.
