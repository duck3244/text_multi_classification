# Korean UnSmile Multi-label Classification

🤖 **KoELECTRA 기반 한국어 혐오 표현 다중 레이블 분류 시스템**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.15%2B-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 프로젝트 개요

이 프로젝트는 **Korean UnSmile 데이터셋**을 활용하여 한국어 텍스트에서 혐오 표현을 자동으로 탐지하고 분류하는 AI 모델을 개발합니다. **KoELECTRA** 모델을 기반으로 하여 높은 성능의 다중 레이블 분류를 수행합니다.

### ✨ 주요 특징

- 🔥 **최신 모델**: KoELECTRA-base-v3-discriminator 사용
- 🏷️ **다중 레이블**: 10개 카테고리 동시 분류
- ⚡ **완전 자동화**: 데이터 처리부터 평가까지 원클릭
- 📊 **상세 분석**: 종합적인 성능 평가 및 시각화
- 🎛️ **유연한 설정**: 다양한 훈련 옵션과 프리셋 제공

### 📋 분류 카테고리

| 카테고리 | 설명 |
|---------|------|
| 👨‍👩‍👧‍👦 **여성/가족** | 성별 관련 편견 및 차별 표현 |
| 👨 **남성** | 남성 집단에 대한 비하 표현 |
| 🏳️‍🌈 **성소수자** | 성소수자 차별 및 혐오 표현 |
| 🌍 **인종/국적** | 특정 인종/국적에 대한 편견 |
| 👶👵 **연령** | 세대 간 갈등 및 연령 차별 |
| 🏙️ **지역** | 지역감정 및 지역 비하 |
| ⛪ **종교** | 종교 혐오 및 차별 표현 |
| 😠 **기타 혐오** | 기타 집단 대상 혐오 표현 |
| 🤬 **악플/욕설** | 일반적인 욕설 및 비속어 |
| ✅ **clean** | 건전한 일반 텍스트 |

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 원클릭 실행

```bash
# 전체 파이프라인 실행 (데이터 처리 → 훈련 → 평가)
python main.py pipeline
```

---

## 📖 상세 사용법

### 🔧 개별 단계 실행

```bash
# 1. 데이터 다운로드 및 전처리
python main.py process-data

# 2. 모델 훈련
python main.py train

# 3. 모델 평가
python main.py evaluate \
    --model_path output_20240628_143022/checkpoints/best_model.pth \
    --eval_data korean_unsmile_csv/korean_unsmile_valid.csv
```

### ⚙️ 고급 옵션

```bash
# GPU 메모리가 부족한 경우
python main.py train --batch_size 4 --max_length 256

# 빠른 테스트
python main.py train --preset debug

# 커스텀 설정
python main.py train \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --dropout_rate 0.2
```

### 📝 설정 파일 사용

```bash
# 설정 파일 생성
python main.py create-config --preset conservative --output_file my_config.json

# 설정 파일로 훈련
python main.py train --config my_config.json
```

---

## 🛠️ 프로젝트 구조

```
korean-unsmile-classification/
├── 📄 main.py                    # 메인 실행 스크립트
├── 📄 data_processor.py          # 데이터 다운로드 및 전처리
├── 📄 model.py                   # 모델 정의
├── 📄 config.py                  # 설정 관리
├── 📄 utils.py                   # 유틸리티 함수
├── 📄 trainer.py                 # 모델 훈련
├── 📄 evaluator.py               # 모델 평가
├── 📄 requirements.txt           # 패키지 요구사항
├── 📄 README.md                  # 프로젝트 가이드
├── 📁 korean_unsmile_csv/        # 처리된 데이터
│   ├── korean_unsmile_train.csv
│   ├── korean_unsmile_valid.csv
│   ├── label_info.json
│   └── class_weights.json
└── 📁 output_YYYYMMDD_HHMMSS/    # 훈련 결과
    ├── 📁 checkpoints/
    │   └── best_model.pth
    ├── 📁 plots/
    │   ├── training_history.png
    │   ├── confusion_matrices.png
    │   └── per_label_performance.png
    ├── experiment_config.json
    └── final_metrics.json
```

---

## 📊 성능 벤치마크

### 🎯 예상 성능 (검증 데이터 기준)

| 메트릭 | 점수 |
|--------|------|
| **Exact Match Accuracy** | ~0.75 |
| **Macro F1** | ~0.80 |
| **Macro Precision** | ~0.82 |
| **Macro Recall** | ~0.78 |
| **Hamming Loss** | ~0.15 |

### ⏱️ 훈련 시간

| 환경 | 시간 |
|------|------|
| **RTX 4090** | ~1.5시간 |
| **RTX 3080** | ~2-3시간 |
| **RTX 4060 Laptop** | ~4-5시간 |
| **Google Colab (T4)** | ~3-4시간 |

---

## 🎛️ 설정 옵션

### 📋 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 권장 범위 |
|---------|--------|------|-----------|
| `batch_size` | 16 | 배치 크기 | 4-32 |
| `learning_rate` | 2e-5 | 학습률 | 1e-5 ~ 5e-5 |
| `num_epochs` | 5 | 훈련 에포크 | 3-10 |
| `max_length` | 512 | 최대 시퀀스 길이 | 128-512 |
| `dropout_rate` | 0.1 | 드롭아웃 비율 | 0.1-0.3 |

### 🎨 사전 정의된 프리셋

| 프리셋 | 용도 | 특징 |
|--------|------|------|
| `default` | 일반 훈련 | 균형잡힌 설정 |
| `debug` | 빠른 테스트 | 작은 데이터, 짧은 훈련 |
| `conservative` | 안정적 훈련 | 낮은 학습률, 높은 드롭아웃 |
| `large_batch` | 고성능 GPU | 큰 배치 크기 |
| `high_lr` | 빠른 수렴 | 높은 학습률 |

---

## 🚨 문제 해결

### 💾 GPU 메모리 부족

```bash
# 해결책 1: 배치 크기 줄이기
python main.py train --batch_size 4

# 해결책 2: 시퀀스 길이 줄이기  
python main.py train --max_length 256

# 해결책 3: 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py train --batch_size 4
```

### 🐛 일반적인 오류

| 오류 | 해결책 |
|------|--------|
| **CUDA out of memory** | `--batch_size 4 --max_length 256` |
| **ModuleNotFoundError** | `pip install -r requirements.txt` |
| **torch.load 오류** | PyTorch 2.6+ 호환성 문제 (자동 해결됨) |
| **데이터셋 다운로드 실패** | 인터넷 연결 확인 후 재시도 |

### ⚡ 성능 최적화

```bash
# Mixed Precision 사용 (실험적)
python main.py train --fp16

# 그래디언트 누적으로 큰 배치 효과
python main.py train --batch_size 4 --gradient_accumulation_steps 4

# 여러 GPU 사용 (구현 예정)
# python main.py train --multi_gpu
```

---

## 📈 결과 분석

훈련 완료 후 다음 결과들이 자동 생성됩니다:

### 📊 시각화 자료
- **훈련 과정 그래프** (`training_history.png`)
- **레이블별 성능 비교** (`per_label_performance.png`) 
- **혼동 행렬** (`confusion_matrices.png`)
- **레이블 분포** (`label_distribution.png`)

### 📄 분석 리포트
- **최종 성능 지표** (`final_metrics.json`)
- **실험 설정** (`experiment_config.json`)
- **상세 예측 결과** (`detailed_predictions.csv`)

### 🔍 해석 가이드

```python
# 결과 해석 예시
{
    "overall": {
        "exact_match_accuracy": 0.75,  # 모든 레이블이 정확한 비율
        "macro_f1": 0.80,              # 평균 F1 점수
        "hamming_loss": 0.15           # 레이블별 오차 평균
    },
    "per_label": {
        "clean": {"f1": 0.95},         # 일반 텍스트 인식률 높음
        "악플/욕설": {"f1": 0.85},      # 욕설 탐지 성능 좋음
        "성소수자": {"f1": 0.65}       # 상대적으로 어려운 카테고리
    }
}
```

---

## 🔬 고급 활용

### 🎯 임계값 최적화

```python
# 평가 스크립트에서 최적 임계값 자동 탐색
python main.py evaluate \
    --model_path best_model.pth \
    --eval_data valid.csv \
    --threshold_analysis
```

### 🧪 실험 관리

```bash
# 여러 실험 병렬 실행
python main.py train --run_name "exp1_small_lr" --learning_rate 1e-5
python main.py train --run_name "exp2_large_batch" --batch_size 32
```

### 📱 실시간 추론

```python
from evaluator import ModelEvaluator

# 모델 로드
evaluator = ModelEvaluator("best_model.pth")

# 단일 텍스트 예측
result = evaluator.predict_single("이 바보야!")
print(result['predicted_labels'])  # ['악플/욕설']
```

### 🔧 개발 환경 설정

```bash
# 개발용 패키지 설치
pip install -r requirements-dev.txt

# 코드 스타일 검사
black . && flake8 .

# 테스트 실행
pytest tests/
```
