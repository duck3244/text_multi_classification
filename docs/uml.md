# UML 다이어그램

본 문서는 **Korean UnSmile Multi-label Classification** 프로젝트의 정적·동적 구조를 Mermaid 기반 UML 로 표현한다. GitHub / GitLab / VS Code Markdown Preview 등에서 그대로 렌더링된다.

---

## 1. 컴포넌트 다이어그램 (Component Diagram)

시스템 전체 컴포넌트의 의존 관계.

```mermaid
flowchart LR
    subgraph Frontend["Frontend (Vue 3 + Vite + TS)"]
        AppVue["App.vue<br/>(form + slider + result + history)"]
        ApiTS["lib/api.ts<br/>(fetch wrapper)"]
        StorageTS["lib/storage.ts<br/>(localStorage)"]
        Vite["vite.config.ts<br/>(/api proxy)"]
    end

    subgraph BackendAPI["Backend API (FastAPI)"]
        FastAPIApp["app.py<br/>(routes + lifespan + CORS)"]
        InferEngine["InferenceEngine<br/>(singleton)"]
        Schemas["schemas.py<br/>(Pydantic I/O)"]
    end

    subgraph BackendCore["Backend Core"]
        Model["model.py<br/>MultiLabelElectraClassifier"]
        Config["config.py<br/>ExperimentConfig"]
        Utils["utils.py<br/>Metrics / Viz / Early stop"]
    end

    subgraph BackendCLI["Backend CLI (Training Plane)"]
        Main["main.py<br/>(CLI entry)"]
        DataProc["data_processor.py<br/>KoreanUnsmileProcessor"]
        Trainer["trainer.py<br/>KoreanUnsmileTrainer"]
        Evaluator["evaluator.py<br/>ModelEvaluator"]
    end

    subgraph External["External"]
        HF[("HuggingFace Hub<br/>smilegate-ai/kor_unsmile<br/>+ KoELECTRA")]
        FS[("filesystem<br/>output/<run>/<br/>best_model.pth")]
    end

    AppVue --> ApiTS
    AppVue --> StorageTS
    ApiTS -->|"/api/*"| Vite
    Vite -.proxy.-> FastAPIApp

    FastAPIApp --> InferEngine
    FastAPIApp --> Schemas
    InferEngine --> Model
    InferEngine --> Config
    InferEngine --> FS

    Main --> DataProc
    Main --> Trainer
    Main --> Evaluator
    Trainer --> Model
    Trainer --> Utils
    Trainer --> Config
    Trainer --> FS
    Evaluator --> Model
    Evaluator --> Utils
    Evaluator --> Config
    DataProc --> HF
    DataProc --> FS
    Model --> HF
```

---

## 2. 클래스 다이어그램 — Backend Core (Model & Config)

KoELECTRA 모델·데이터셋·손실 함수·실험 설정.

```mermaid
classDiagram
    class KoreanUnsmileDataset {
        +List~str~ texts
        +List~ListInt~ labels
        +ElectraTokenizer tokenizer
        +int max_length
        +__len__() int
        +__getitem__(idx) Dict
    }

    class MultiLabelElectraClassifier {
        +ElectraModel electra
        +ElectraConfig config
        +int num_labels
        +Dropout dropout
        +Linear classifier
        +_init_weights()
        +forward(input_ids, attention_mask, labels) Dict
        +predict(input_ids, attention_mask, threshold) Tuple
        +freeze_electra()
        +unfreeze_electra()
    }

    class WeightedBCEWithLogitsLoss {
        +Tensor pos_weights
        +str reduction
        +forward(logits, labels) Tensor
    }

    class FocalLoss {
        +float alpha
        +float gamma
        +str reduction
        +forward(logits, labels) Tensor
    }

    class ModelConfig {
        +str model_name
        +int num_labels = 10
        +int max_length = 512
        +float dropout_rate = 0.1
        +str loss_type
        +float focal_alpha
        +float focal_gamma
    }

    class TrainingConfig {
        +int batch_size
        +int num_epochs
        +float learning_rate
        +float weight_decay
        +float warmup_ratio
        +str scheduler_type
        +int eval_steps
        +float threshold
        +int early_stopping_patience
        +int gradient_accumulation_steps
        +bool fp16
    }

    class DataConfig {
        +str data_dir
        +str train_file
        +str valid_file
        +str text_column
        +List label_columns
        +int max_samples
    }

    class LoggingConfig {
        +str output_dir
        +str run_name
        +bool use_wandb
        +bool use_tensorboard
        +bool save_predictions
    }

    class ExperimentConfig {
        +ModelConfig model
        +TrainingConfig training
        +DataConfig data
        +LoggingConfig logging
        +str experiment_name
        +int seed = 42
        +__post_init__()
        +save(path)
        +load(path) ExperimentConfig$
        +update(**kwargs)
        +to_dict() Dict
    }

    class ConfigManager {
        +Dict presets
        +create_config(preset, **overrides) ExperimentConfig
        -_create_default_config()
        -_create_debug_config()
        -_create_large_batch_config()
        -_create_high_lr_config()
        -_create_conservative_config()
    }

    class TorchDataset {
        <<external>>
    }
    class NnModule {
        <<external>>
    }
    KoreanUnsmileDataset --|> TorchDataset
    MultiLabelElectraClassifier --|> NnModule
    WeightedBCEWithLogitsLoss --|> NnModule
    FocalLoss --|> NnModule

    ExperimentConfig *-- ModelConfig
    ExperimentConfig *-- TrainingConfig
    ExperimentConfig *-- DataConfig
    ExperimentConfig *-- LoggingConfig
    ConfigManager ..> ExperimentConfig : creates
```

---

## 3. 클래스 다이어그램 — Training & Evaluation Pipeline

데이터 처리·학습 루프·평가·유틸리티.

```mermaid
classDiagram
    class KoreanUnsmileProcessor {
        +str output_dir
        +List label_columns
        +Dict label_descriptions
        +download_dataset() Dict
        +process_split(data, split_name) DataFrame
        +calculate_statistics(df) Dict
        +calculate_class_weights(df) Dict
        +save_data_files(train, valid, all)
        +save_metadata(train, valid, all)
        +process() Tuple
    }

    class KoreanUnsmileTrainer {
        +ExperimentConfig config
        +torch.device device
        +Logger logger
        +TrainingHistory history
        +EarlyStopping early_stopping
        +Dict best_metrics
        +load_data() Tuple
        +create_model_and_optimizer(...) Tuple
        -_build_criterion() nn.Module
        +calculate_loss(logits, labels) Tensor
        +train_epoch(model, loader, opt, sched) float
        +evaluate(model, loader, labels) Dict
        +check_early_stopping(metrics, model) bool
        +save_results(final_metrics)
        +create_visualizations(final_metrics)
        +train() Dict
    }

    class ModelEvaluator {
        +torch.device device
        +ExperimentConfig config
        +Dict checkpoint
        +List label_columns
        +MultiLabelElectraClassifier model
        +load_label_info()
        +load_model() MultiLabelElectraClassifier
        +create_data_loader(file, batch_size) Tuple
        +predict(loader) Tuple
        +evaluate_predictions(preds, labels, t) Dict
        +analyze_threshold_sensitivity(...) Tuple
        +analyze_per_label_performance(...) DataFrame
        +analyze_prediction_distribution(...) Dict
        +generate_classification_report(...) str
        +find_difficult_samples(...) List
        +create_evaluation_report(...) Dict
    }

    class MetricsCalculator {
        +List label_names
        +float threshold
        +calculate_metrics(y_true, y_pred, y_scores) Dict
        -_calculate_overall_metrics(...) Dict
        -_calculate_per_label_metrics(...) Dict
        -_calculate_confusion_matrices(...) List
    }

    class EarlyStopping {
        +int patience
        +float min_delta
        +str mode
        +bool restore_best_weights
        +int counter
        +float best_score
        +bool early_stop
        +Dict best_weights
        +__call__(score, model) bool
        -_is_improvement(score) bool
        -_save_checkpoint(model)
    }

    class TrainingHistory {
        +Dict history
        +update(**kwargs)
        +save(path)
        +load(path)
        +get_best_metric(name, mode) Tuple
    }

    class VisualizationUtils {
        <<utility>>
        +plot_training_history(history, path)$
        +plot_label_distribution(counts, path)$
        +plot_confusion_matrices(cms, names, path)$
        +plot_per_label_metrics(metrics, path)$
    }

    KoreanUnsmileTrainer --> ExperimentConfig
    KoreanUnsmileTrainer --> MultiLabelElectraClassifier : creates
    KoreanUnsmileTrainer --> KoreanUnsmileDataset : creates
    KoreanUnsmileTrainer --> MetricsCalculator : uses
    KoreanUnsmileTrainer --> EarlyStopping : uses
    KoreanUnsmileTrainer --> TrainingHistory : uses
    KoreanUnsmileTrainer --> VisualizationUtils : uses

    ModelEvaluator --> ExperimentConfig
    ModelEvaluator --> MultiLabelElectraClassifier : loads
    ModelEvaluator --> MetricsCalculator : uses
    ModelEvaluator --> VisualizationUtils : uses
```

---

## 4. 클래스 다이어그램 — API (Serving Plane)

```mermaid
classDiagram
    class FastAPIApp {
        <<FastAPI>>
        +state.engine: InferenceEngine
        +lifespan(app)
        +health() HealthResponse
        +labels() LabelsResponse
        +predict(req) PredictResponse
    }

    class InferenceEngine {
        +str model_checkpoint_path
        +str label_info_path
        +torch.device device
        +List label_columns
        +Dict label_descriptions
        +str model_name
        +int max_length
        +MultiLabelElectraClassifier model
        +ElectraTokenizer tokenizer
        +bool loaded
        +load()
        -_load_label_info()
        -_load_checkpoint_and_model()
        +predict(text, threshold) Dict
    }

    class PredictRequest {
        +str text
        +float threshold
    }

    class LabelScore {
        +str name
        +float probability
        +bool predicted
    }

    class PredictResponse {
        +str text
        +float threshold
        +List~LabelScore~ labels
        +LabelScore top
    }

    class LabelMeta {
        +str name
        +str description
    }

    class LabelsResponse {
        +List~LabelMeta~ labels
    }

    class HealthResponse {
        +str status
        +bool model_loaded
        +str device
    }

    FastAPIApp --> InferenceEngine
    FastAPIApp ..> PredictRequest
    FastAPIApp ..> PredictResponse
    FastAPIApp ..> LabelsResponse
    FastAPIApp ..> HealthResponse
    InferenceEngine --> MultiLabelElectraClassifier
    PredictResponse *-- LabelScore
    LabelsResponse *-- LabelMeta
```

---

## 5. 클래스 다이어그램 — Frontend

```mermaid
classDiagram
    class AppVue {
        <<VueSFC>>
        +Ref~string~ text
        +Ref~number~ threshold
        +Ref~boolean~ loading
        +Ref~string~ error
        +Ref~PredictResponse~ result
        +Ref~HealthResponse~ health
        +Ref~ArrayHistoryItem~ history
        +Computed sortedLabels
        +submit() async
        +reuseHistory(item)
        +onClearHistory()
        +pct(p) string
        +onMounted()
    }

    class api {
        <<TSmodule>>
        +health() Promise~HealthResponse~
        +labels() Promise~LabelsList~
        +predict(text, threshold) Promise~PredictResponse~
    }

    class storage {
        <<TSmodule>>
        +KEY "unsmile.history.v1"
        +MAX 10
        +loadHistory() HistoryItem[]
        +pushHistory(result) HistoryItem[]
        +clearHistory()
    }

    class PredictResponse {
        +string text
        +number threshold
        +LabelScore[] labels
        +LabelScore top
    }

    class LabelScore {
        +string name
        +number probability
        +boolean predicted
    }

    class HistoryItem {
        +number at
        +PredictResponse result
    }

    AppVue --> api : uses
    AppVue --> storage : uses
    api ..> PredictResponse
    storage ..> HistoryItem
    HistoryItem *-- PredictResponse
    PredictResponse *-- LabelScore
```

---

## 6. 시퀀스 다이어그램 — 학습 파이프라인

`python backend/main.py pipeline` 실행 시 호출 흐름.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as main.py
    participant Cfg as ConfigManager
    participant DP as KoreanUnsmileProcessor
    participant HF as HuggingFace Hub
    participant FS as filesystem
    participant TR as KoreanUnsmileTrainer
    participant Model as MultiLabelElectraClassifier
    participant Eval as ModelEvaluator

    User->>CLI: python main.py pipeline
    CLI->>Cfg: create_config(preset)
    Cfg-->>CLI: ExperimentConfig

    CLI->>DP: process()
    DP->>HF: load_dataset("smilegate-ai/kor_unsmile")
    HF-->>DP: train/valid splits
    DP->>DP: process_split / calculate_class_weights
    DP->>FS: write CSV + label_info.json + class_weights.json

    CLI->>TR: KoreanUnsmileTrainer(config).train()
    TR->>FS: read CSV + label_info + class_weights
    TR->>Model: build & to(device)
    loop epochs
        TR->>Model: forward(batch)
        Model-->>TR: logits
        TR->>TR: BCE/Weighted/Focal loss
        TR->>Model: backward + optimizer.step
        opt evaluate at eval_steps
            TR->>Model: forward(valid)
            TR->>TR: MetricsCalculator.calculate_metrics
            alt is best
                TR->>FS: save best_model.pth
            end
            alt patience exceeded
                TR-->>CLI: early stop
            end
        end
    end
    TR->>FS: training_history.json, final_metrics.json, plots/

    CLI->>Eval: ModelEvaluator(best_model.pth)
    Eval->>FS: load checkpoint + label_info
    Eval->>Model: load_state_dict
    Eval->>Eval: predict + threshold sweep + per-label + difficult samples
    Eval->>FS: evaluation_report.json + plots/
    Eval-->>CLI: report
    CLI-->>User: 성공 메시지 + 출력 경로
```

---

## 7. 시퀀스 다이어그램 — 추론 요청

브라우저에서 분류 버튼을 누른 순간부터 결과 표시까지.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant V as App.vue
    participant Api as lib/api.ts
    participant Vite as Vite dev proxy
    participant App as FastAPI app
    participant Eng as InferenceEngine
    participant Tok as ElectraTokenizer
    participant M as MultiLabelElectraClassifier
    participant LS as localStorage

    U->>V: 텍스트 + threshold 입력, 분류 클릭
    V->>Api: predict(text, threshold)
    Api->>Vite: POST /api/predict {text, threshold}
    Vite->>App: forward → :8000
    App->>App: PredictRequest 검증 (1≤len≤4000, 0≤t≤1)
    App->>Eng: run_in_threadpool(engine.predict)
    Eng->>Tok: tokenizer(text, max_length=512)
    Tok-->>Eng: input_ids, attention_mask
    Eng->>M: forward(input_ids, attention_mask) @inference_mode
    M-->>Eng: logits [1, 10]
    Eng->>Eng: sigmoid → 10 probs → labels + top
    Eng-->>App: dict
    App-->>Api: PredictResponse (JSON)
    Api-->>V: PredictResponse
    V->>V: sortedLabels (확률 내림차순), 막대 차트
    V->>LS: pushHistory(result) (max 10건)
    V-->>U: 결과 표시
```

---

## 8. 시퀀스 다이어그램 — 서버 부팅 (lifespan)

FastAPI 가 뜨면서 모델을 1회 로드하는 과정.

```mermaid
sequenceDiagram
    autonumber
    participant Uvicorn
    participant App as FastAPI
    participant Eng as InferenceEngine
    participant FS as filesystem
    participant HF as HF Cache
    participant M as MultiLabelElectraClassifier

    Uvicorn->>App: startup
    App->>App: lifespan() __aenter__
    App->>Eng: InferenceEngine()
    App->>Eng: load()
    Eng->>FS: read label_info.json
    Eng->>FS: torch.load(best_model.pth)
    Eng->>M: MultiLabelElectraClassifier(model_name, num_labels=10)
    M->>HF: from_pretrained (cached)
    M-->>Eng: instance
    Eng->>M: load_state_dict + to(device) + eval()
    Eng->>HF: ElectraTokenizer.from_pretrained
    Eng-->>App: loaded = True
    App->>App: app.state.engine = engine
    Note over App: yield (서비스 시작)
```

---

## 9. 상태 다이어그램 — Early Stopping

`utils.py:EarlyStopping` 의 상태 전이.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Tracking : 첫 score 도착<br/>best_score = score
    Tracking --> Tracking : score 개선<br/>counter=0, 가중치 저장
    Tracking --> Waiting : 개선 없음<br/>counter += 1
    Waiting --> Tracking : score 개선<br/>counter=0
    Waiting --> Waiting : 개선 없음 (counter < patience)
    Waiting --> Stopped : counter >= patience<br/>best weights 복원
    Stopped --> [*]
```

---

## 10. 활동 다이어그램 — 학습 한 에포크 (Gradient Accumulation 포함)

```mermaid
flowchart TD
    Start([Epoch 시작])
    Init[optimizer.zero_grad / total_loss=0]
    Loop{batch 남음?}
    Forward["model.forward<br/>logits = f(x)"]
    Loss["loss = criterion(logits, y)<br/>loss /= grad_accum"]
    Backward[loss.backward]
    Accum{batch_idx+1 mod<br/>grad_accum == 0?}
    Clip[clip_grad_norm_]
    Step[optimizer.step<br/>scheduler.step<br/>zero_grad / step += 1]
    LogCheck{step mod logging_steps == 0?}
    Log[logger.info Loss/LR]
    EvalCheck{eval_strategy=steps and<br/>step mod eval_steps == 0?}
    Eval[evaluate on valid<br/>check_early_stopping]
    Stop{early_stop?}
    Done([Epoch 종료<br/>avg loss 반환])
    Early([조기 반환])

    Start --> Init --> Loop
    Loop -- yes --> Forward --> Loss --> Backward --> Accum
    Accum -- no --> Loop
    Accum -- yes --> Clip --> Step --> LogCheck
    LogCheck -- yes --> Log --> EvalCheck
    LogCheck -- no --> EvalCheck
    EvalCheck -- yes --> Eval --> Stop
    EvalCheck -- no --> Loop
    Stop -- yes --> Early
    Stop -- no --> Loop
    Loop -- no --> Done
```

---

## 11. ER 다이어그램 (간이) — 산출물 파일 관계

학습/평가가 생성하는 파일 간의 의미 관계.

```mermaid
erDiagram
    EXPERIMENT_CONFIG ||--|| BEST_MODEL : "produces"
    EXPERIMENT_CONFIG ||--|| TRAINING_HISTORY : "logs"
    EXPERIMENT_CONFIG ||--|| FINAL_METRICS : "measures"
    BEST_MODEL ||--|{ EVALUATION_REPORT : "evaluated by"
    LABEL_INFO ||--o{ EVALUATION_REPORT : "labels"
    LABEL_INFO ||--o{ FINAL_METRICS : "labels"
    CLASS_WEIGHTS ||--|| EXPERIMENT_CONFIG : "loaded into"
    CSV_TRAIN ||--|| BEST_MODEL : "trains"
    CSV_VALID ||--|| EVALUATION_REPORT : "evaluated on"

    EXPERIMENT_CONFIG {
        string model_name
        int batch_size
        float learning_rate
        int num_epochs
        string loss_type
        int seed
    }
    BEST_MODEL {
        dict model_state_dict
        dict optimizer_state_dict
        int epoch
        int step
        dict metrics
        string timestamp
    }
    LABEL_INFO {
        list label_columns
        int num_labels
        dict label_descriptions
    }
    CLASS_WEIGHTS {
        dict label_to_pos_weight
    }
    TRAINING_HISTORY {
        list train_loss
        list eval_loss
        list eval_metrics
        list learning_rates
    }
    FINAL_METRICS {
        float exact_match_accuracy
        float macro_f1
        float hamming_loss
        dict per_label
    }
    EVALUATION_REPORT {
        dict overall_metrics
        dict per_label_metrics
        list threshold_analysis
        list difficult_samples
    }
```

---

## 12. 패키지 다이어그램

```mermaid
flowchart TB
    subgraph backend
        direction TB
        config[config.py]
        model[model.py]
        utils[utils.py]
        data_processor[data_processor.py]
        trainer[trainer.py]
        evaluator[evaluator.py]
        main[main.py]
        subgraph api
            app[app.py]
            inference[inference.py]
            schemas[schemas.py]
        end
    end

    subgraph frontend
        direction TB
        AppVue2[App.vue]
        subgraph lib
            apiTs[api.ts]
            storageTs[storage.ts]
        end
        viteCfg[vite.config.ts]
    end

    main --> trainer
    main --> evaluator
    main --> data_processor
    main --> config
    trainer --> model
    trainer --> utils
    trainer --> config
    evaluator --> model
    evaluator --> utils
    evaluator --> config
    data_processor --> config
    inference --> model
    inference --> config
    app --> inference
    app --> schemas

    AppVue2 --> apiTs
    AppVue2 --> storageTs
    apiTs -.HTTP via vite proxy.-> app
```

---

## 13. 다이어그램 인덱스

| # | 다이어그램 | 범위 |
|---|---|---|
| 1 | 컴포넌트 다이어그램 | 전체 시스템 |
| 2 | 클래스 다이어그램 | Model & Config |
| 3 | 클래스 다이어그램 | Training & Eval |
| 4 | 클래스 다이어그램 | API (Serving) |
| 5 | 클래스 다이어그램 | Frontend |
| 6 | 시퀀스 다이어그램 | 학습 파이프라인 |
| 7 | 시퀀스 다이어그램 | 추론 요청 |
| 8 | 시퀀스 다이어그램 | 서버 부팅 |
| 9 | 상태 다이어그램 | Early Stopping |
| 10 | 활동 다이어그램 | 학습 1 에포크 |
| 11 | ER 다이어그램 | 산출물 파일 |
| 12 | 패키지 다이어그램 | 모듈 의존 관계 |
