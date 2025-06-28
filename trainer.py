"""
trainer.py
KoELECTRA 다중 레이블 분류 모델 훈련
"""

import os
import json
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from torch.optim import AdamW  # PyTorch에서 직접 import
from torch.utils.data import DataLoader
from transformers import (
    ElectraTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# 로컬 모듈 임포트
from model import MultiLabelElectraClassifier, KoreanUnsmileDataset, create_model
from config import ExperimentConfig
from utils import (
    MetricsCalculator, EarlyStopping, TrainingHistory, VisualizationUtils,
    set_seed, setup_logging, create_output_directories, save_json,
    format_metrics_for_logging, save_model_checkpoint, calculate_class_weights
)


class KoreanUnsmileTrainer:
    """Korean UnSmile 분류 모델 훈련 클래스"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 시드 설정
        set_seed(config.seed)

        # 출력 디렉토리 생성
        create_output_directories(config.logging.output_dir)

        # 로깅 설정
        log_file = os.path.join(config.logging.output_dir, 'logs', 'training.log')
        self.logger = setup_logging(log_file)

        # 훈련 기록
        self.history = TrainingHistory()
        self.step = 0
        self.epoch = 0

        # 조기 종료
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_threshold,
            mode='max' if 'f1' in config.training.early_stopping_metric else 'min'
        )

        # 최고 성능 기록
        self.best_metrics = {}

        self.logger.info(f"훈련 디바이스: {self.device}")
        self.logger.info(f"설정: {config.experiment_name}")

    def load_data(self) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, float]]:
        """데이터 로드 및 데이터로더 생성"""

        self.logger.info("데이터 로딩 중...")

        # 데이터 파일 경로
        train_file = os.path.join(self.config.data.data_dir, self.config.data.train_file)
        valid_file = os.path.join(self.config.data.data_dir, self.config.data.valid_file)

        # CSV 파일 로드
        import pandas as pd
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)

        # 레이블 정보 로드
        label_info_file = os.path.join(self.config.data.data_dir, 'label_info.json')
        with open(label_info_file, 'r', encoding='utf-8') as f:
            label_info = json.load(f)

        label_columns = label_info['label_columns']
        self.config.model.num_labels = len(label_columns)

        # 클래스 가중치 로드
        class_weights_file = os.path.join(self.config.data.data_dir, 'class_weights.json')
        with open(class_weights_file, 'r', encoding='utf-8') as f:
            class_weights = json.load(f)

        # 최대 샘플 수 제한 (디버깅용)
        if self.config.data.max_samples:
            train_df = train_df.head(self.config.data.max_samples)
            valid_df = valid_df.head(self.config.data.max_samples // 5)

        # 텍스트와 레이블 추출
        train_texts = train_df[self.config.data.text_column].astype(str).tolist()
        valid_texts = valid_df[self.config.data.text_column].astype(str).tolist()

        train_labels = train_df[label_columns].values.tolist()
        valid_labels = valid_df[label_columns].values.tolist()

        self.logger.info(f"학습 데이터: {len(train_texts):,}개")
        self.logger.info(f"검증 데이터: {len(valid_texts):,}개")
        self.logger.info(f"레이블 수: {len(label_columns)}개")

        # 토크나이저 로드
        tokenizer = ElectraTokenizer.from_pretrained(self.config.model.model_name)

        # 데이터셋 생성
        train_dataset = KoreanUnsmileDataset(
            train_texts, train_labels, tokenizer, self.config.model.max_length
        )
        valid_dataset = KoreanUnsmileDataset(
            valid_texts, valid_labels, tokenizer, self.config.model.max_length
        )

        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.dataloader_pin_memory
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.dataloader_pin_memory
        )

        return train_loader, valid_loader, label_columns, class_weights

    def create_model_and_optimizer(self, label_columns: List[str],
                                  class_weights: Dict[str, float]) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
        """모델, 옵티마이저, 스케줄러 생성"""

        self.logger.info("모델 생성 중...")

        # 모델 생성
        model = MultiLabelElectraClassifier(
            model_name=self.config.model.model_name,
            num_labels=self.config.model.num_labels,
            dropout_rate=self.config.model.dropout_rate,
            hidden_size=self.config.model.hidden_size
        )

        model.to(self.device)

        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"전체 파라미터: {total_params:,}")
        self.logger.info(f"훈련 가능한 파라미터: {trainable_params:,}")

        # 옵티마이저 생성
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            eps=1e-8
        )

        # 스케줄러 생성
        num_training_steps = len(self.train_loader) * self.config.training.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)

        if self.config.training.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.training.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None

        # 클래스 가중치 처리
        if class_weights:
            weights_list = [class_weights.get(label, 1.0) for label in label_columns]
            self.class_weights = torch.FloatTensor(weights_list).to(self.device)
            self.logger.info(f"클래스 가중치 적용: {weights_list}")
        else:
            self.class_weights = None

        return model, optimizer, scheduler

    def calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""

        if self.class_weights is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        return loss_fct(logits, labels)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, scheduler) -> float:
        """한 에포크 훈련"""

        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1} Training")

        for batch_idx, batch in enumerate(progress_bar):
            # 배치를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 그래디언트 누적을 위한 처리
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # 순전파
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            # 그래디언트 누적을 위한 손실 정규화
            loss = loss / self.config.training.gradient_accumulation_steps
            total_loss += loss.item()

            # 역전파
            loss.backward()

            # 그래디언트 누적 단계마다 옵티마이저 스텝
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)

                # 옵티마이저 스텝
                optimizer.step()
                if scheduler:
                    scheduler.step()

                self.step += 1

                # 학습률 기록
                current_lr = scheduler.get_last_lr()[0] if scheduler else self.config.training.learning_rate
                self.history.update(learning_rates=current_lr, steps=self.step)

                # 로깅
                if self.step % self.config.logging.logging_steps == 0:
                    self.logger.info(
                        f"Step {self.step}: Loss={loss.item() * self.config.training.gradient_accumulation_steps:.4f}, "
                        f"LR={current_lr:.2e}"
                    )

                # 중간 평가
                if (self.config.training.eval_strategy == "steps" and
                    self.step % self.config.training.eval_steps == 0):
                    eval_metrics = self.evaluate(model, self.valid_loader, self.label_columns)

                    # 조기 종료 확인
                    if self.check_early_stopping(eval_metrics, model):
                        return total_loss / num_batches

            # 프로그레스 바 업데이트
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.training.gradient_accumulation_steps:.4f}'
            })

        return total_loss / num_batches

    def evaluate(self, model: nn.Module, data_loader: DataLoader,
                label_columns: List[str]) -> Dict[str, Any]:
        """모델 평가"""

        model.eval()

        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs['loss']
                logits = outputs['logits']

                total_loss += loss.item()

                # 확률로 변환
                probabilities = torch.sigmoid(logits)

                all_predictions.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        # 결과 합치기
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        all_logits = np.vstack(all_logits)

        # 메트릭 계산
        metrics_calculator = MetricsCalculator(label_columns, self.config.training.threshold)
        metrics = metrics_calculator.calculate_metrics(all_labels, all_predictions, all_predictions)

        # 평균 손실 추가
        metrics['eval_loss'] = total_loss / len(data_loader)

        # 훈련 기록 업데이트
        self.history.update(eval_loss=metrics['eval_loss'], eval_metrics=metrics)

        # 로깅
        self.logger.info(f"평가 완료 - Step {self.step}")
        self.logger.info(f"  Eval Loss: {metrics['eval_loss']:.4f}")
        self.logger.info(f"  Exact Match Accuracy: {metrics['overall']['exact_match_accuracy']:.4f}")
        self.logger.info(f"  Macro F1: {metrics['overall']['macro_f1']:.4f}")
        self.logger.info(f"  Hamming Loss: {metrics['overall']['hamming_loss']:.4f}")

        model.train()
        return metrics

    def check_early_stopping(self, metrics: Dict[str, Any], model: nn.Module) -> bool:
        """조기 종료 확인 및 베스트 모델 저장"""

        # 메트릭 선택
        metric_name = self.config.training.early_stopping_metric
        if metric_name == "eval_loss":
            score = metrics['eval_loss']
        elif metric_name == "eval_f1":
            score = metrics['overall']['macro_f1']
        elif metric_name == "eval_accuracy":
            score = metrics['overall']['exact_match_accuracy']
        else:
            score = metrics['overall']['macro_f1']  # 기본값

        # 조기 종료 확인
        should_stop = self.early_stopping(score, model)

        # 베스트 모델인지 확인
        is_best = False
        if not self.best_metrics or self._is_better_score(score, metric_name):
            self.best_metrics = metrics.copy()
            is_best = True

            # 베스트 모델 저장
            best_model_path = os.path.join(
                self.config.logging.output_dir,
                'checkpoints',
                'best_model.pth'
            )
            save_model_checkpoint(
                model, self.optimizer, self.scheduler,
                self.epoch, self.step, metrics, best_model_path,
                self.config.to_dict()
            )
            self.logger.info(f"새로운 베스트 모델 저장! {metric_name}: {score:.4f}")

        # 정기 체크포인트 저장
        if self.step % self.config.training.save_steps == 0:
            checkpoint_path = os.path.join(
                self.config.logging.output_dir,
                'checkpoints',
                f'checkpoint_step_{self.step}.pth'
            )
            save_model_checkpoint(
                model, self.optimizer, self.scheduler,
                self.epoch, self.step, metrics, checkpoint_path,
                self.config.to_dict()
            )

        return should_stop

    def _is_better_score(self, score: float, metric_name: str) -> bool:
        """더 좋은 점수인지 확인"""
        if not self.best_metrics:
            return True

        if metric_name == "eval_loss":
            previous_score = self.best_metrics['eval_loss']
            return score < previous_score
        elif metric_name == "eval_f1":
            previous_score = self.best_metrics['overall']['macro_f1']
            return score > previous_score
        elif metric_name == "eval_accuracy":
            previous_score = self.best_metrics['overall']['exact_match_accuracy']
            return score > previous_score
        else:
            return False

    def save_results(self, final_metrics: Dict[str, Any]) -> None:
        """결과 저장"""

        output_dir = self.config.logging.output_dir

        # 훈련 기록 저장
        history_path = os.path.join(output_dir, 'training_history.json')
        self.history.save(history_path)

        # 최종 메트릭 저장
        metrics_path = os.path.join(output_dir, 'final_metrics.json')
        save_json(final_metrics, metrics_path)

        # 베스트 메트릭 저장
        if self.best_metrics:
            best_metrics_path = os.path.join(output_dir, 'best_metrics.json')
            save_json(self.best_metrics, best_metrics_path)

        # 설정 저장
        config_path = os.path.join(output_dir, 'experiment_config.json')
        self.config.save(config_path)

        self.logger.info(f"결과 저장 완료: {output_dir}")

    def create_visualizations(self, final_metrics: Dict[str, Any]) -> None:
        """시각화 생성"""

        output_dir = self.config.logging.output_dir
        plots_dir = os.path.join(output_dir, 'plots')

        # 훈련 과정 시각화
        if self.history.history['train_loss'] or self.history.history['eval_metrics']:
            training_plot_path = os.path.join(plots_dir, 'training_history.png')
            VisualizationUtils.plot_training_history(self.history, training_plot_path)

        # 레이블 분포 시각화
        if 'per_label' in final_metrics:
            label_counts = {
                label: int(metrics['support'])
                for label, metrics in final_metrics['per_label'].items()
            }
            distribution_plot_path = os.path.join(plots_dir, 'label_distribution.png')
            VisualizationUtils.plot_label_distribution(label_counts, distribution_plot_path)

        # 레이블별 메트릭 시각화
        if 'per_label' in final_metrics:
            metrics_plot_path = os.path.join(plots_dir, 'per_label_metrics.png')
            VisualizationUtils.plot_per_label_metrics(final_metrics['per_label'], metrics_plot_path)

        # 혼동 행렬 시각화
        if 'confusion_matrices' in final_metrics:
            confusion_plot_path = os.path.join(plots_dir, 'confusion_matrices.png')
            VisualizationUtils.plot_confusion_matrices(
                final_metrics['confusion_matrices'],
                self.label_columns,
                confusion_plot_path
            )

        self.logger.info(f"시각화 저장 완료: {plots_dir}")

    def train(self) -> Dict[str, Any]:
        """전체 훈련 프로세스"""

        self.logger.info("훈련 시작!")
        start_time = time.time()

        # 데이터 로드
        self.train_loader, self.valid_loader, self.label_columns, class_weights = self.load_data()

        # 모델 및 옵티마이저 생성
        model, self.optimizer, self.scheduler = self.create_model_and_optimizer(
            self.label_columns, class_weights
        )

        self.logger.info(f"총 에포크: {self.config.training.num_epochs}")
        self.logger.info(f"배치 크기: {self.config.training.batch_size}")
        self.logger.info(f"학습률: {self.config.training.learning_rate}")
        self.logger.info(f"총 훈련 스텝: {len(self.train_loader) * self.config.training.num_epochs}")

        # 훈련 루프
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch

            self.logger.info(f"\n=== Epoch {epoch + 1}/{self.config.training.num_epochs} ===")

            # 한 에포크 훈련
            train_loss = self.train_epoch(model, self.train_loader, self.optimizer, self.scheduler)
            self.history.update(train_loss=train_loss, epochs=epoch + 1)

            self.logger.info(f"Epoch {epoch + 1} 완료 - 평균 손실: {train_loss:.4f}")

            # 에포크 단위 평가
            if self.config.training.eval_strategy == "epoch":
                eval_metrics = self.evaluate(model, self.valid_loader, self.label_columns)

                # 조기 종료 확인
                if self.check_early_stopping(eval_metrics, model):
                    self.logger.info(f"조기 종료! Epoch {epoch + 1}에서 훈련 중단")
                    break

        # 최종 평가
        self.logger.info("\n=== 최종 평가 ===")

        # 베스트 모델 로드
        best_model_path = os.path.join(
            self.config.logging.output_dir, 'checkpoints', 'best_model.pth'
        )
        if os.path.exists(best_model_path):
            try:
                # PyTorch 2.6+ 호환성을 위해 weights_only=False 사용
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            except TypeError:
                # 구버전 PyTorch에서는 weights_only 파라미터가 없음
                checkpoint = torch.load(best_model_path, map_location=self.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("베스트 모델 로드됨")

        final_metrics = self.evaluate(model, self.valid_loader, self.label_columns)

        # 훈련 시간 계산
        training_time = time.time() - start_time
        final_metrics['training_time_seconds'] = training_time
        final_metrics['training_time_formatted'] = f"{training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.0f}s"

        # 결과 출력
        self.logger.info(f"훈련 완료! 총 시간: {final_metrics['training_time_formatted']}")
        self.logger.info(f"최종 정확도: {final_metrics['overall']['exact_match_accuracy']:.4f}")
        self.logger.info(f"최종 Macro F1: {final_metrics['overall']['macro_f1']:.4f}")
        self.logger.info(f"최종 Hamming Loss: {final_metrics['overall']['hamming_loss']:.4f}")

        # 레이블별 결과
        self.logger.info("\n=== 레이블별 최종 결과 ===")
        for label, metrics in final_metrics['per_label'].items():
            self.logger.info(
                f"{label}: F1={metrics['f1']:.3f}, "
                f"Precision={metrics['precision']:.3f}, "
                f"Recall={metrics['recall']:.3f}"
            )

        # 결과 저장
        self.save_results(final_metrics)

        # 시각화 생성
        if self.config.logging.save_metrics:
            self.create_visualizations(final_metrics)

        return final_metrics


def main():
    """메인 함수"""
    import argparse
    from config import create_config_from_args, validate_config

    parser = argparse.ArgumentParser(description="Korean UnSmile Multi-label Classification Training")

    # 설정 관련
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--preset", type=str, default="default",
                       choices=["default", "debug", "large_batch", "high_lr", "conservative"],
                       help="사전 정의된 설정")

    # 모델 관련
    parser.add_argument("--model_name", type=str, default="monologg/koelectra-base-v3-discriminator")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # 훈련 관련
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 데이터 관련
    parser.add_argument("--data_dir", type=str, default="korean_unsmile_csv")
    parser.add_argument("--max_samples", type=int, help="최대 샘플 수 (디버깅용)")

    # 로깅 관련
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--use_wandb", action="store_true", help="WandB 사용")

    # 기타
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 설정 생성
    if args.config and os.path.exists(args.config):
        from config import ExperimentConfig
        config = ExperimentConfig.load(args.config)
    else:
        config = create_config_from_args(args)

    # 설정 검증
    try:
        validate_config(config)
    except ValueError as e:
        print(f"설정 오류: {e}")
        return

    # 트레이너 생성 및 훈련
    trainer = KoreanUnsmileTrainer(config)

    try:
        final_metrics = trainer.train()
        print("\n훈련이 성공적으로 완료되었습니다!")
        print(f"결과는 {config.logging.output_dir}에 저장되었습니다.")

    except Exception as e:
        trainer.logger.error(f"훈련 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()