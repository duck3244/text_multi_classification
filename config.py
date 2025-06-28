"""
config.py
훈련 설정 및 하이퍼파라미터 관리
"""

import json
import os

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    model_name: str = "monologg/koelectra-base-v3-discriminator"
    num_labels: int = 10
    max_length: int = 512
    dropout_rate: float = 0.1
    hidden_size: Optional[int] = None
    
    # 손실 함수 설정
    loss_type: str = "weighted_bce"  # "bce", "weighted_bce", "focal"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 스케줄러 설정
    scheduler_type: str = "linear"  # "linear", "cosine", "polynomial"
    
    # 평가 관련
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch"
    threshold: float = 0.5
    
    # 저장 관련
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # 조기 종료
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    early_stopping_metric: str = "eval_f1"  # "eval_loss", "eval_f1", "eval_accuracy"
    
    # 기타
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    fp16: bool = False  # Mixed precision training
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    data_dir: str = "korean_unsmile_csv"
    train_file: str = "korean_unsmile_train.csv"
    valid_file: str = "korean_unsmile_valid.csv"
    test_file: Optional[str] = None
    
    # 데이터 전처리
    text_column: str = "text"
    label_columns: Optional[list] = None
    max_samples: Optional[int] = None  # 디버깅용
    
    # 데이터 증강
    use_data_augmentation: bool = False
    augmentation_rate: float = 0.1


@dataclass
class LoggingConfig:
    """로깅 관련 설정"""
    output_dir: str = "output"
    run_name: Optional[str] = None
    logging_steps: int = 100
    
    # WandB 설정
    use_wandb: bool = False
    wandb_project: str = "korean-unsmile-classification"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None
    
    # 텐서보드 설정
    use_tensorboard: bool = True
    
    # 저장할 메트릭
    save_metrics: bool = True
    save_predictions: bool = True
    save_attention_weights: bool = False


@dataclass
class ExperimentConfig:
    """전체 실험 설정"""
    # 기본 설정들
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    
    # 실험 메타데이터
    experiment_name: str = "koelectra_multilabel"
    description: str = ""
    tags: Optional[list] = None
    
    # 재현성
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 출력 디렉토리에 타임스탬프 추가
        if not self.logging.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logging.run_name = f"{self.experiment_name}_{timestamp}"
            self.logging.output_dir = os.path.join(self.logging.output_dir, self.logging.run_name)
    
    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """JSON 파일에서 설정 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 중첩된 설정 객체들 생성
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        data_config = DataConfig(**config_dict['data'])
        logging_config = LoggingConfig(**config_dict['logging'])
        
        # 메인 설정 객체 생성
        config = cls(
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config
        )
        
        # 나머지 필드들 설정
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'logging']:
                setattr(config, key, value)
        
        return config
    
    def update(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif '.' in key:
                # 중첩된 설정 업데이트 (예: "model.learning_rate")
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def get_model_name(self) -> str:
        """모델명 반환 (파일명에 사용)"""
        return self.model.model_name.replace('/', '_')


class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self):
        self.presets = {
            'default': self._create_default_config,
            'debug': self._create_debug_config,
            'large_batch': self._create_large_batch_config,
            'high_lr': self._create_high_lr_config,
            'conservative': self._create_conservative_config
        }
    
    def create_config(self, preset: str = 'default', **overrides) -> ExperimentConfig:
        """사전 정의된 설정으로 config 생성"""
        
        if preset not in self.presets:
            raise ValueError(f"알 수 없는 preset: {preset}. 사용 가능한 preset: {list(self.presets.keys())}")
        
        config = self.presets[preset]()
        config.update(**overrides)
        
        return config
    
    def _create_default_config(self) -> ExperimentConfig:
        """기본 설정"""
        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            logging=LoggingConfig(),
            experiment_name="koelectra_default"
        )
    
    def _create_debug_config(self) -> ExperimentConfig:
        """디버깅용 설정 (빠른 실행)"""
        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(
                batch_size=4,
                num_epochs=1,
                eval_steps=50,
                save_steps=100,
                early_stopping_patience=1
            ),
            data=DataConfig(max_samples=1000),
            logging=LoggingConfig(logging_steps=10),
            experiment_name="koelectra_debug"
        )
    
    def _create_large_batch_config(self) -> ExperimentConfig:
        """큰 배치 크기 설정"""
        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(
                batch_size=32,
                learning_rate=3e-5,
                gradient_accumulation_steps=2
            ),
            data=DataConfig(),
            logging=LoggingConfig(),
            experiment_name="koelectra_large_batch"
        )
    
    def _create_high_lr_config(self) -> ExperimentConfig:
        """높은 학습률 설정"""
        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_ratio=0.2,
                weight_decay=0.1
            ),
            data=DataConfig(),
            logging=LoggingConfig(),
            experiment_name="koelectra_high_lr"
        )
    
    def _create_conservative_config(self) -> ExperimentConfig:
        """보수적인 설정 (안정적인 훈련)"""
        return ExperimentConfig(
            model=ModelConfig(dropout_rate=0.2),
            training=TrainingConfig(
                batch_size=8,
                learning_rate=1e-5,
                num_epochs=10,
                warmup_ratio=0.15,
                early_stopping_patience=5
            ),
            data=DataConfig(),
            logging=LoggingConfig(),
            experiment_name="koelectra_conservative"
        )


def create_config_from_args(args) -> ExperimentConfig:
    """명령행 인자에서 설정 생성"""
    
    config_manager = ConfigManager()
    
    # 기본 설정 또는 프리셋 사용
    preset = getattr(args, 'preset', 'default')
    config = config_manager.create_config(preset)
    
    # 명령행 인자로 오버라이드
    overrides = {}
    
    # 모델 관련
    if hasattr(args, 'model_name'):
        overrides['model.model_name'] = args.model_name
    if hasattr(args, 'max_length'):
        overrides['model.max_length'] = args.max_length
    if hasattr(args, 'dropout_rate'):
        overrides['model.dropout_rate'] = args.dropout_rate
    
    # 훈련 관련
    if hasattr(args, 'batch_size'):
        overrides['training.batch_size'] = args.batch_size
    if hasattr(args, 'learning_rate'):
        overrides['training.learning_rate'] = args.learning_rate
    if hasattr(args, 'num_epochs'):
        overrides['training.num_epochs'] = args.num_epochs
    if hasattr(args, 'weight_decay'):
        overrides['training.weight_decay'] = args.weight_decay
    
    # 데이터 관련
    if hasattr(args, 'data_dir'):
        overrides['data.data_dir'] = args.data_dir
    if hasattr(args, 'max_samples'):
        overrides['data.max_samples'] = args.max_samples
    
    # 로깅 관련
    if hasattr(args, 'output_dir'):
        overrides['logging.output_dir'] = args.output_dir
    if hasattr(args, 'use_wandb'):
        overrides['logging.use_wandb'] = args.use_wandb
    
    # 기타
    if hasattr(args, 'seed'):
        overrides['seed'] = args.seed
    
    config.update(**overrides)
    
    return config


def validate_config(config: ExperimentConfig) -> None:
    """설정 유효성 검사"""
    
    errors = []
    
    # 모델 설정 검사
    if config.model.num_labels <= 0:
        errors.append("num_labels는 0보다 커야 합니다")
    
    if config.model.max_length <= 0:
        errors.append("max_length는 0보다 커야 합니다")
    
    if not 0 <= config.model.dropout_rate <= 1:
        errors.append("dropout_rate는 0과 1 사이여야 합니다")
    
    # 훈련 설정 검사
    if config.training.batch_size <= 0:
        errors.append("batch_size는 0보다 커야 합니다")
    
    if config.training.learning_rate <= 0:
        errors.append("learning_rate는 0보다 커야 합니다")
    
    if config.training.num_epochs <= 0:
        errors.append("num_epochs는 0보다 커야 합니다")
    
    if not 0 <= config.training.warmup_ratio <= 1:
        errors.append("warmup_ratio는 0과 1 사이여야 합니다")
    
    # 데이터 설정 검사
    if not os.path.exists(config.data.data_dir):
        errors.append(f"데이터 디렉토리가 존재하지 않습니다: {config.data.data_dir}")
    
    # 에러가 있으면 예외 발생
    if errors:
        raise ValueError("설정 검증 실패:\n" + "\n".join(f"- {error}" for error in errors))


# 사용 예제
if __name__ == "__main__":
    # 설정 매니저 생성
    config_manager = ConfigManager()
    
    # 기본 설정 생성
    config = config_manager.create_config('default')
    
    # 설정 출력
    print("기본 설정:")
    print(f"  모델: {config.model.model_name}")
    print(f"  배치 크기: {config.training.batch_size}")
    print(f"  학습률: {config.training.learning_rate}")
    print(f"  에포크: {config.training.num_epochs}")
    
    # 설정 저장 및 로드 테스트
    config.save("test_config.json")
    loaded_config = ExperimentConfig.load("test_config.json")
    
    print("\n로드된 설정이 동일한지 확인:", config.to_dict() == loaded_config.to_dict())
    
    # 다른 프리셋 테스트
    debug_config = config_manager.create_config('debug')
    print(f"\n디버그 설정 배치 크기: {debug_config.training.batch_size}")
    print(f"디버그 설정 최대 샘플: {debug_config.data.max_samples}")
    
    # 설정 검증
    try:
        validate_config(config)
        print("\n설정 검증 통과")
    except ValueError as e:
        print(f"\n설정 검증 실패: {e}")
    
    # 정리
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
