"""
utils.py
훈련 및 평가에 필요한 유틸리티 함수들
"""

import os
import json
import torch
import random
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    hamming_loss, multilabel_confusion_matrix, roc_auc_score
)


def set_seed(seed: int = 42) -> None:
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """로깅 설정"""
    
    # 로거 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_output_directories(output_dir: str) -> None:
    """출력 디렉토리들 생성"""
    
    directories = [
        output_dir,
        os.path.join(output_dir, 'checkpoints'),
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'plots'),
        os.path.join(output_dir, 'predictions'),
        os.path.join(output_dir, 'tensorboard')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_json(data: Any, file_path: str) -> None:
    """JSON 파일 저장"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(file_path: str) -> Any:
    """JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class MetricsCalculator:
    """다중 레이블 분류 메트릭 계산 클래스"""
    
    def __init__(self, label_names: List[str], threshold: float = 0.5):
        self.label_names = label_names
        self.threshold = threshold
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """종합적인 메트릭 계산"""
        
        # 이진 예측으로 변환
        if y_scores is not None:
            y_pred_binary = (y_scores > self.threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
        
        metrics = {}
        
        # 전체 메트릭
        metrics['overall'] = self._calculate_overall_metrics(y_true, y_pred_binary, y_scores)
        
        # 레이블별 메트릭
        metrics['per_label'] = self._calculate_per_label_metrics(y_true, y_pred_binary, y_scores)
        
        # 혼동 행렬
        metrics['confusion_matrices'] = self._calculate_confusion_matrices(y_true, y_pred_binary)
        
        return metrics
    
    def _calculate_per_label_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """레이블별 메트릭 계산"""

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_label_metrics = {}

        for i, label_name in enumerate(self.label_names):
            metrics = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
                'accuracy': float((y_pred[:, i] == y_true[:, i]).mean()),
                'positive_count': int(y_true[:, i].sum()),
                'predicted_count': int(y_pred[:, i].sum()),
                'true_positive': int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()),
                'false_positive': int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()),
                'false_negative': int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()),
                'true_negative': int(((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum())
            }

            # Specificity (True Negative Rate) 계산
            tn = metrics['true_negative']
            fp = metrics['false_positive']
            if (tn + fp) > 0:
                metrics['specificity'] = float(tn / (tn + fp))
            else:
                metrics['specificity'] = 0.0

            # AUC 추가 (가능한 경우)
            if y_scores is not None and len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                    metrics['auc'] = float(auc)
                except:
                    metrics['auc'] = 0.5
            else:
                metrics['auc'] = None

            per_label_metrics[label_name] = metrics

        return per_label_metrics

    def _calculate_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[np.ndarray]:
        """혼동 행렬 계산"""
        return multilabel_confusion_matrix(y_true, y_pred).tolist()

    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """전체 메트릭 계산"""

        metrics = {}

        # Exact Match Accuracy (모든 레이블이 정확히 일치)
        metrics['exact_match_accuracy'] = np.all(y_pred == y_true, axis=1).mean()

        # Hamming Loss (레이블별 불일치 평균)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)

        # Subset Accuracy (정확히 일치하는 비율)
        metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)

        # Macro/Micro 평균
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics['macro_precision'] = precision.mean()
        metrics['macro_recall'] = recall.mean()
        metrics['macro_f1'] = f1.mean()

        # Micro 평균
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )

        metrics['micro_precision'] = micro_precision
        metrics['micro_recall'] = micro_recall
        metrics['micro_f1'] = micro_f1

        # AUC 계산 (가능한 경우)
        if y_scores is not None:
            try:
                auc_scores = []
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) > 1:
                        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                        auc_scores.append(auc)
                    else:
                        auc_scores.append(0.5)

                metrics['macro_auc'] = np.mean(auc_scores)
                metrics['auc_scores'] = auc_scores

            except Exception as e:
                print(f"AUC 계산 오류: {e}")
                metrics['macro_auc'] = None

        return metrics

    def _calculate_per_label_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """레이블별 메트릭 계산"""

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_label_metrics = {}

        for i, label_name in enumerate(self.label_names):
            metrics = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
                'accuracy': float((y_pred[:, i] == y_true[:, i]).mean()),
                'positive_count': int(y_true[:, i].sum()),
                'predicted_count': int(y_pred[:, i].sum()),
                'true_positive': int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()),
                'false_positive': int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()),
                'false_negative': int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()),
                'true_negative': int(((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum())
            }

            # Specificity (True Negative Rate) 계산
            tn = metrics['true_negative']
            fp = metrics['false_positive']
            if (tn + fp) > 0:
                metrics['specificity'] = float(tn / (tn + fp))
            else:
                metrics['specificity'] = 0.0

            # AUC 추가 (가능한 경우)
            if y_scores is not None and len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                    metrics['auc'] = float(auc)
                except:
                    metrics['auc'] = 0.5
            else:
                metrics['auc'] = None

            per_label_metrics[label_name] = metrics

        return per_label_metrics

    def _calculate_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[np.ndarray]:
        """혼동 행렬 계산"""
        return multilabel_confusion_matrix(y_true, y_pred).tolist()


class EarlyStopping:
    """조기 종료 클래스"""

    def __init__(self, patience: int = 3, min_delta: float = 0.001,
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Args:
            patience: 개선되지 않는 에포크 수
            min_delta: 최소 개선 폭
            mode: 'min' (손실) 또는 'max' (정확도, F1 등)
            restore_best_weights: 최고 성능 가중치 복원 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """조기 종료 확인"""

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        """개선 여부 확인"""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def _save_checkpoint(self, model: torch.nn.Module) -> None:
        """체크포인트 저장"""
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


class TrainingHistory:
    """훈련 기록 클래스"""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_metrics': [],
            'learning_rates': [],
            'epochs': [],
            'steps': []
        }

    def update(self, **kwargs):
        """기록 업데이트"""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)

    def save(self, file_path: str):
        """기록 저장"""
        save_json(self.history, file_path)

    def load(self, file_path: str):
        """기록 로드"""
        self.history = load_json(file_path)

    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """최고 성능 메트릭과 에포크 반환"""
        if metric_name not in self.history or not self.history[metric_name]:
            return None, None

        values = self.history[metric_name]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        return values[best_idx], best_idx


class VisualizationUtils:
    """시각화 유틸리티 클래스"""

    @staticmethod
    def plot_training_history(history: TrainingHistory, save_path: str = None,
                            figsize: Tuple[int, int] = (15, 10)):
        """훈련 과정 시각화"""

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training History', fontsize=16)

        # 손실 그래프
        if history.history['train_loss']:
            axes[0, 0].plot(history.history['train_loss'], label='Train Loss')
        if history.history['eval_loss']:
            axes[0, 0].plot(history.history['eval_loss'], label='Eval Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Steps/Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 학습률 그래프
        if history.history['learning_rates']:
            axes[0, 1].plot(history.history['learning_rates'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)

        # 메트릭 그래프들
        if history.history['eval_metrics']:
            # F1 점수
            f1_scores = [m.get('overall', {}).get('macro_f1', 0) for m in history.history['eval_metrics']]
            axes[0, 2].plot(f1_scores)
            axes[0, 2].set_title('Macro F1 Score')
            axes[0, 2].set_xlabel('Evaluation Steps')
            axes[0, 2].set_ylabel('F1 Score')
            axes[0, 2].grid(True)

            # 정확도
            accuracies = [m.get('overall', {}).get('exact_match_accuracy', 0) for m in history.history['eval_metrics']]
            axes[1, 0].plot(accuracies)
            axes[1, 0].set_title('Exact Match Accuracy')
            axes[1, 0].set_xlabel('Evaluation Steps')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].grid(True)

            # Hamming Loss
            hamming_losses = [m.get('overall', {}).get('hamming_loss', 0) for m in history.history['eval_metrics']]
            axes[1, 1].plot(hamming_losses)
            axes[1, 1].set_title('Hamming Loss')
            axes[1, 1].set_xlabel('Evaluation Steps')
            axes[1, 1].set_ylabel('Hamming Loss')
            axes[1, 1].grid(True)

            # AUC
            auc_scores = [m.get('overall', {}).get('macro_auc', 0) for m in history.history['eval_metrics'] if m.get('overall', {}).get('macro_auc') is not None]
            if auc_scores:
                axes[1, 2].plot(auc_scores)
                axes[1, 2].set_title('Macro AUC')
                axes[1, 2].set_xlabel('Evaluation Steps')
                axes[1, 2].set_ylabel('AUC')
                axes[1, 2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_label_distribution(label_counts: Dict[str, int], save_path: str = None,
                              figsize: Tuple[int, int] = (12, 8)):
        """레이블 분포 시각화"""

        labels = list(label_counts.keys())
        counts = list(label_counts.values())

        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(labels)), counts)
        plt.title('Label Distribution', fontsize=16)
        plt.xlabel('Labels', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

        # 막대 위에 수치 표시
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(counts),
                    str(count), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_confusion_matrices(confusion_matrices: List[np.ndarray], label_names: List[str],
                              save_path: str = None, figsize: Tuple[int, int] = (20, 16)):
        """혼동 행렬들 시각화"""

        n_labels = len(label_names)
        n_cols = 4
        n_rows = (n_labels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Confusion Matrices per Label', fontsize=16)

        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        for i, (cm, label) in enumerate(zip(confusion_matrices, label_names)):
            cm = np.array(cm)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Pred 0', 'Pred 1'],
                       yticklabels=['True 0', 'True 1'])
            axes[i].set_title(f'{label}')

        # 빈 서브플롯 숨기기
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_per_label_metrics(per_label_metrics: Dict[str, Dict[str, float]],
                             save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """레이블별 메트릭 시각화"""

        labels = list(per_label_metrics.keys())
        metrics = ['precision', 'recall', 'f1']

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Per-Label Metrics', fontsize=16)

        for i, metric in enumerate(metrics):
            values = [per_label_metrics[label][metric] for label in labels]

            bars = axes[i].bar(range(len(labels)), values)
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_xlabel('Labels')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_xticks(range(len(labels)))
            axes[i].set_xticklabels(labels, rotation=45, ha='right')
            axes[i].set_ylim(0, 1)

            # 막대 위에 수치 표시
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def calculate_class_weights(labels: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """클래스 불균형을 위한 가중치 계산"""

    if method == 'balanced':
        n_samples = len(labels)
        n_classes = labels.shape[1]

        weights = []
        for i in range(n_classes):
            n_positive = labels[:, i].sum()
            n_negative = n_samples - n_positive

            if n_positive == 0:
                weight = 1.0
            else:
                weight = n_negative / n_positive

            weights.append(weight)

        return np.array(weights)

    elif method == 'inverse_frequency':
        frequencies = labels.mean(axis=0)
        weights = 1.0 / (frequencies + 1e-6)
        return weights / weights.sum() * len(weights)

    else:
        return np.ones(labels.shape[1])


def format_metrics_for_logging(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """로깅을 위한 메트릭 포맷팅"""

    formatted = {}

    # 전체 메트릭
    if 'overall' in metrics:
        for key, value in metrics['overall'].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted[f"{prefix}overall_{key}"] = float(value)

    # 레이블별 메트릭 (요약)
    if 'per_label' in metrics:
        # 평균 계산
        all_precision = [m['precision'] for m in metrics['per_label'].values()]
        all_recall = [m['recall'] for m in metrics['per_label'].values()]
        all_f1 = [m['f1'] for m in metrics['per_label'].values()]

        formatted[f"{prefix}avg_precision"] = float(np.mean(all_precision))
        formatted[f"{prefix}avg_recall"] = float(np.mean(all_recall))
        formatted[f"{prefix}avg_f1"] = float(np.mean(all_f1))

    return formatted


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         scheduler, epoch: int, step: int, metrics: Dict[str, Any],
                         save_path: str, config: Dict[str, Any] = None) -> None:
    """모델 체크포인트 저장"""

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_model_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer = None,
                         scheduler = None) -> Dict[str, Any]:
    """모델 체크포인트 로드"""

    try:
        # PyTorch 2.6+ 호환성을 위해 weights_only=False 사용
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 구버전 PyTorch에서는 weights_only 파라미터가 없음
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


# 사용 예제
if __name__ == "__main__":
    # 시드 설정
    set_seed(42)

    # 로깅 설정
    logger = setup_logging("test.log")
    logger.info("테스트 시작")

    # 가짜 데이터로 메트릭 계산 테스트
    label_names = ['label1', 'label2', 'label3']
    y_true = np.random.randint(0, 2, (100, 3))
    y_scores = np.random.random((100, 3))

    # 메트릭 계산기
    calculator = MetricsCalculator(label_names)
    metrics = calculator.calculate_metrics(y_true, y_scores, y_scores)

    print("메트릭 계산 완료:")
    print(f"Exact Match Accuracy: {metrics['overall']['exact_match_accuracy']:.4f}")
    print(f"Macro F1: {metrics['overall']['macro_f1']:.4f}")

    # 시각화 테스트
    history = TrainingHistory()
    history.update(train_loss=0.5, eval_loss=0.4, eval_metrics=metrics)

    # 출력 디렉토리 생성
    create_output_directories("test_output")

    print("유틸리티 함수 테스트 완료!")

    # 정리
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
    if os.path.exists("test.log"):
        os.remove("test.log")

    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """전체 메트릭 계산"""

        metrics = {}

        # Exact Match Accuracy (모든 레이블이 정확히 일치)
        metrics['exact_match_accuracy'] = np.all(y_pred == y_true, axis=1).mean()

        # Hamming Loss (레이블별 불일치 평균)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)

        # Subset Accuracy (정확히 일치하는 비율)
        metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)

        # Macro/Micro 평균
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics['macro_precision'] = precision.mean()
        metrics['macro_recall'] = recall.mean()
        metrics['macro_f1'] = f1.mean()

        # Micro 평균
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )

        metrics['micro_precision'] = micro_precision
        metrics['micro_recall'] = micro_recall
        metrics['micro_f1'] = micro_f1

        # AUC 계산 (가능한 경우)
        if y_scores is not None:
            try:
                auc_scores = []
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) > 1:
                        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                        auc_scores.append(auc)
                    else:
                        auc_scores.append(0.5)

                metrics['macro_auc'] = np.mean(auc_scores)
                metrics['auc_scores'] = auc_scores

            except Exception as e:
                print(f"AUC 계산 오류: {e}")
                metrics['macro_auc'] = None

        return metrics

    def _calculate_per_label_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """레이블별 메트릭 계산"""

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_label_metrics = {}

        for i, label_name in enumerate(self.label_names):
            metrics = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
                'accuracy': float((y_pred[:, i] == y_true[:, i]).mean()),
                'positive_count': int(y_true[:, i].sum()),
                'predicted_count': int(y_pred[:, i].sum()),
                'true_positive': int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()),
                'false_positive': int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()),
                'false_negative': int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()),
                'true_negative': int(((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum())
            }

            # Specificity (True Negative Rate) 계산
            tn = metrics['true_negative']
            fp = metrics['false_positive']
            if (tn + fp) > 0:
                metrics['specificity'] = float(tn / (tn + fp))
            else:
                metrics['specificity'] = 0.0

            # AUC 추가 (가능한 경우)
            if y_scores is not None and len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                    metrics['auc'] = float(auc)
                except:
                    metrics['auc'] = 0.5
            else:
                metrics['auc'] = None

            per_label_metrics[label_name] = metrics

        return per_label_metrics