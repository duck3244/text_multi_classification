"""
evaluator.py
훈련된 모델의 성능 평가 및 분석
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional, Any

# 로컬 모듈 임포트
from model import MultiLabelElectraClassifier, KoreanUnsmileDataset
from utils import MetricsCalculator, VisualizationUtils, setup_logging, save_json
from config import ExperimentConfig


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Args:
            model_path: 훈련된 모델 체크포인트 경로
            config_path: 설정 파일 경로
            device: 사용할 디바이스
        """
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 로깅 설정
        self.logger = setup_logging()
        
        self.logger.info(f"평가 디바이스: {self.device}")
        self.logger.info(f"모델 경로: {model_path}")
        
        # 체크포인트 로드
        try:
            # PyTorch 2.6+ 호환성을 위해 weights_only=False 사용
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # 구버전 PyTorch에서는 weights_only 파라미터가 없음
            self.checkpoint = torch.load(model_path, map_location=self.device)

        # 설정 로드
        if config_path and os.path.exists(config_path):
            from config import ExperimentConfig
            self.config = ExperimentConfig.load(config_path)
        elif 'config' in self.checkpoint:
            # 체크포인트에서 설정 로드
            config_dict = self.checkpoint['config']
            self.config = self._dict_to_config(config_dict)
        else:
            # 기본 설정 사용
            self.config = self._create_default_config()

        # 레이블 정보 로드
        self.load_label_info()

        # 모델 로드
        self.model = self.load_model()

        self.logger.info("모델 평가기 초기화 완료")

    def _create_default_config(self):
        """기본 설정 생성"""
        from config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, LoggingConfig

        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            logging=LoggingConfig()
        )

    def _dict_to_config(self, config_dict: Dict) -> ExperimentConfig:
        """딕셔너리를 설정 객체로 변환"""
        from config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, LoggingConfig

        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))

        config = ExperimentConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config
        )

        # 나머지 필드 설정
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'logging'] and hasattr(config, key):
                setattr(config, key, value)

        return config

    def load_label_info(self):
        """레이블 정보 로드"""

        label_info_path = os.path.join(self.config.data.data_dir, 'label_info.json')

        if os.path.exists(label_info_path):
            with open(label_info_path, 'r', encoding='utf-8') as f:
                label_info = json.load(f)

            self.label_columns = label_info['label_columns']
            self.label_descriptions = label_info.get('label_descriptions', {})
        else:
            # 기본 레이블 사용
            self.label_columns = [
                '여성/가족', '남성', '성소수자', '인종/국적', '연령',
                '지역', '종교', '기타 혐오', '악플/욕설', 'clean'
            ]
            self.label_descriptions = {}

        self.config.model.num_labels = len(self.label_columns)

        self.logger.info(f"레이블 수: {len(self.label_columns)}")

    def load_model(self) -> MultiLabelElectraClassifier:
        """모델 로드"""

        model = MultiLabelElectraClassifier(
            model_name=self.config.model.model_name,
            num_labels=self.config.model.num_labels,
            dropout_rate=self.config.model.dropout_rate,
            hidden_size=self.config.model.hidden_size
        )

        # 가중치 로드
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # 모델 정보
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"모델 파라미터 수: {total_params:,}")

        return model

    def create_data_loader(self, data_file: str, batch_size: int = 32) -> Tuple[DataLoader, List[str], List[List[int]]]:
        """데이터로더 생성"""

        # CSV 파일 로드
        df = pd.read_csv(data_file)

        texts = df[self.config.data.text_column].astype(str).tolist()

        # 레이블이 있는 경우
        if all(col in df.columns for col in self.label_columns):
            labels = df[self.label_columns].values.tolist()
        else:
            # 레이블이 없는 경우 (예측만 수행)
            labels = [[0] * len(self.label_columns) for _ in range(len(texts))]

        # 토크나이저 로드
        from transformers import ElectraTokenizer
        tokenizer = ElectraTokenizer.from_pretrained(self.config.model.model_name)

        # 데이터셋 생성
        dataset = KoreanUnsmileDataset(
            texts, labels, tokenizer, self.config.model.max_length
        )

        # 데이터로더 생성
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return data_loader, texts, labels

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측 수행"""

        self.logger.info("예측 수행 중...")

        all_predictions = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']

                # 확률로 변환
                probabilities = torch.sigmoid(logits)

                all_predictions.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        # 결과 합치기
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        logits = np.vstack(all_logits)

        self.logger.info(f"예측 완료: {len(predictions)} 샘플")

        return predictions, labels, logits

    def evaluate_predictions(self, predictions: np.ndarray, labels: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, Any]:
        """예측 결과 평가"""

        # 메트릭 계산
        metrics_calculator = MetricsCalculator(self.label_columns, threshold)
        metrics = metrics_calculator.calculate_metrics(labels, predictions, predictions)

        return metrics

    def analyze_threshold_sensitivity(self, predictions: np.ndarray, labels: np.ndarray,
                                    thresholds: List[float] = None) -> Dict[str, List[float]]:
        """임계값 민감도 분석"""

        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1).tolist()

        self.logger.info("임계값 민감도 분석 중...")

        results = {
            'thresholds': thresholds,
            'exact_match_accuracy': [],
            'macro_f1': [],
            'macro_precision': [],
            'macro_recall': [],
            'hamming_loss': []
        }

        for threshold in thresholds:
            metrics_calculator = MetricsCalculator(self.label_columns, threshold)
            metrics = metrics_calculator.calculate_metrics(labels, predictions, predictions)

            results['exact_match_accuracy'].append(metrics['overall']['exact_match_accuracy'])
            results['macro_f1'].append(metrics['overall']['macro_f1'])
            results['macro_precision'].append(metrics['overall']['macro_precision'])
            results['macro_recall'].append(metrics['overall']['macro_recall'])
            results['hamming_loss'].append(metrics['overall']['hamming_loss'])

        # 최적 임계값 찾기
        best_f1_idx = np.argmax(results['macro_f1'])
        best_threshold = thresholds[best_f1_idx]

        self.logger.info(f"최적 임계값 (Macro F1 기준): {best_threshold:.2f}")

        return results, best_threshold

    def analyze_per_label_performance(self, predictions: np.ndarray, labels: np.ndarray,
                                    threshold: float = 0.5) -> pd.DataFrame:
        """레이블별 성능 상세 분석"""

        metrics_calculator = MetricsCalculator(self.label_columns, threshold)
        metrics = metrics_calculator.calculate_metrics(labels, predictions, predictions)

        # DataFrame으로 변환
        per_label_data = []
        for label, label_metrics in metrics['per_label'].items():
            row = {
                'Label': label,
                'Precision': label_metrics['precision'],
                'Recall': label_metrics['recall'],
                'F1': label_metrics['f1'],
                'Support': label_metrics['support'],
                'Accuracy': label_metrics['accuracy'],
                'Positive_Count': label_metrics['positive_count'],
                'Predicted_Count': label_metrics['predicted_count'],
                'True_Positive': label_metrics['true_positive'],
                'False_Positive': label_metrics['false_positive'],
                'False_Negative': label_metrics['false_negative'],
                'True_Negative': label_metrics['true_negative'],
                'Specificity': label_metrics['specificity']
            }

            if label_metrics['auc'] is not None:
                row['AUC'] = label_metrics['auc']
            else:
                row['AUC'] = 0.5

            # 레이블 설명 추가
            if label in self.label_descriptions:
                row['Description'] = self.label_descriptions[label]
            else:
                row['Description'] = ""

            per_label_data.append(row)

        df = pd.DataFrame(per_label_data)
        df = df.sort_values('F1', ascending=False)

        return df

    def analyze_prediction_distribution(self, predictions: np.ndarray,
                                      threshold: float = 0.5) -> Dict[str, Any]:
        """예측 분포 분석"""

        # 이진 예측으로 변환
        binary_predictions = (predictions > threshold).astype(int)

        # 샘플별 예측된 레이블 수
        labels_per_sample = binary_predictions.sum(axis=1)

        # 레이블별 예측 빈도
        label_predictions = binary_predictions.sum(axis=0)

        # 가장 자주 예측되는 레이블 조합
        unique_combinations, combination_counts = np.unique(
            binary_predictions, axis=0, return_counts=True
        )

        # 상위 10개 조합
        top_combinations_idx = np.argsort(combination_counts)[-10:][::-1]
        top_combinations = []

        for idx in top_combinations_idx:
            combination = unique_combinations[idx]
            count = combination_counts[idx]
            predicted_labels = [self.label_columns[i] for i in range(len(combination)) if combination[i] == 1]

            top_combinations.append({
                'labels': predicted_labels if predicted_labels else ['clean'],
                'count': int(count),
                'percentage': count / len(predictions) * 100
            })

        analysis = {
            'total_samples': len(predictions),
            'avg_labels_per_sample': float(labels_per_sample.mean()),
            'max_labels_per_sample': int(labels_per_sample.max()),
            'samples_with_no_labels': int((labels_per_sample == 0).sum()),
            'label_prediction_counts': {
                self.label_columns[i]: int(count)
                for i, count in enumerate(label_predictions)
            },
            'top_label_combinations': top_combinations
        }

        return analysis

    def generate_classification_report(self, predictions: np.ndarray, labels: np.ndarray,
                                     threshold: float = 0.5) -> str:
        """분류 리포트 생성"""

        binary_predictions = (predictions > threshold).astype(int)

        # sklearn의 classification_report 사용
        report = classification_report(
            labels, binary_predictions,
            target_names=self.label_columns,
            zero_division=0
        )

        return report

    def find_difficult_samples(self, predictions: np.ndarray, labels: np.ndarray,
                             texts: List[str], n_samples: int = 20) -> List[Dict[str, Any]]:
        """어려운 샘플 찾기 (낮은 확신도 또는 잘못 예측된 샘플)"""

        # 예측 확신도 계산 (0.5에서 얼마나 멀리 떨어져 있는지)
        confidence_scores = np.abs(predictions - 0.5).mean(axis=1)

        # 정확도 계산 (샘플별로 모든 레이블이 정확한지)
        binary_predictions = (predictions > 0.5).astype(int)
        sample_accuracy = (binary_predictions == labels).all(axis=1)

        # 어려운 샘플 기준
        # 1. 낮은 확신도
        # 2. 잘못 예측된 샘플
        difficulty_scores = []

        for i in range(len(predictions)):
            score = 0

            # 확신도가 낮으면 점수 증가
            score += (1 - confidence_scores[i]) * 0.5

            # 잘못 예측되면 점수 증가
            if not sample_accuracy[i]:
                score += 0.5

            difficulty_scores.append(score)

        # 가장 어려운 샘플들 선택
        difficult_indices = np.argsort(difficulty_scores)[-n_samples:][::-1]

        difficult_samples = []
        for idx in difficult_indices:
            predicted_labels = [
                self.label_columns[i] for i in range(len(self.label_columns))
                if binary_predictions[idx, i] == 1
            ]
            true_labels = [
                self.label_columns[i] for i in range(len(self.label_columns))
                if labels[idx, i] == 1
            ]

            sample_info = {
                'index': int(idx),
                'text': texts[idx],
                'true_labels': true_labels if true_labels else ['clean'],
                'predicted_labels': predicted_labels if predicted_labels else ['clean'],
                'confidence_score': float(confidence_scores[idx]),
                'is_correct': bool(sample_accuracy[idx]),
                'difficulty_score': float(difficulty_scores[idx]),
                'label_probabilities': {
                    self.label_columns[i]: float(predictions[idx, i])
                    for i in range(len(self.label_columns))
                }
            }

            difficult_samples.append(sample_info)

        return difficult_samples

    def create_evaluation_report(self, data_file: str, output_dir: str,
                               threshold: float = 0.5, batch_size: int = 32) -> Dict[str, Any]:
        """종합 평가 리포트 생성"""

        self.logger.info(f"평가 리포트 생성: {data_file}")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 데이터 로드 및 예측
        data_loader, texts, labels = self.create_data_loader(data_file, batch_size)
        predictions, labels_array, logits = self.predict(data_loader)

        # 기본 메트릭 계산
        metrics = self.evaluate_predictions(predictions, labels_array, threshold)

        # 임계값 민감도 분석
        threshold_analysis, best_threshold = self.analyze_threshold_sensitivity(
            predictions, labels_array
        )

        # 레이블별 성능 분석
        per_label_df = self.analyze_per_label_performance(predictions, labels_array, threshold)

        # 예측 분포 분석
        distribution_analysis = self.analyze_prediction_distribution(predictions, threshold)

        # 분류 리포트
        classification_report_str = self.generate_classification_report(
            predictions, labels_array, threshold
        )

        # 어려운 샘플 찾기
        difficult_samples = self.find_difficult_samples(
            predictions, labels_array, texts, n_samples=20
        )

        # 종합 리포트
        evaluation_report = {
            'model_info': {
                'model_path': str(self.checkpoint.get('model_path', 'unknown')),
                'model_name': self.config.model.model_name,
                'num_labels': len(self.label_columns),
                'threshold': threshold,
                'best_threshold': best_threshold
            },
            'dataset_info': {
                'data_file': data_file,
                'total_samples': len(texts),
                'label_columns': self.label_columns
            },
            'overall_metrics': metrics['overall'],
            'per_label_metrics': metrics['per_label'],
            'threshold_analysis': threshold_analysis,
            'distribution_analysis': distribution_analysis,
            'difficult_samples': difficult_samples
        }

        # 결과 저장
        # 1. JSON 리포트
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        save_json(evaluation_report, report_path)

        # 2. 레이블별 성능 CSV
        per_label_csv_path = os.path.join(output_dir, 'per_label_performance.csv')
        per_label_df.to_csv(per_label_csv_path, index=False, encoding='utf-8')

        # 3. 분류 리포트 텍스트
        classification_report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(classification_report_path, 'w', encoding='utf-8') as f:
            f.write(classification_report_str)

        # 4. 예측 결과 CSV
        predictions_df = pd.DataFrame({
            'text': texts,
            **{f'{label}_prob': predictions[:, i] for i, label in enumerate(self.label_columns)},
            **{f'{label}_pred': (predictions[:, i] > threshold).astype(int) for i, label in enumerate(self.label_columns)},
            **{f'{label}_true': labels_array[:, i] for i, label in enumerate(self.label_columns)}
        })

        predictions_csv_path = os.path.join(output_dir, 'detailed_predictions.csv')
        predictions_df.to_csv(predictions_csv_path, index=False, encoding='utf-8')

        # 시각화 생성
        self.create_evaluation_visualizations(
            evaluation_report, predictions, labels_array, output_dir
        )

        self.logger.info(f"평가 리포트 저장 완료: {output_dir}")

        return evaluation_report

    def create_evaluation_visualizations(self, report: Dict[str, Any],
                                       predictions: np.ndarray, labels: np.ndarray,
                                       output_dir: str):
        """평가 시각화 생성"""

        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 1. 임계값 민감도 플롯
        self.plot_threshold_sensitivity(report['threshold_analysis'], plots_dir)

        # 2. 레이블별 성능 플롯
        self.plot_per_label_performance(report['per_label_metrics'], plots_dir)

        # 3. 혼동 행렬
        if 'confusion_matrices' in report:
            VisualizationUtils.plot_confusion_matrices(
                report['confusion_matrices'], self.label_columns,
                os.path.join(plots_dir, 'confusion_matrices.png')
            )

        # 4. 예측 분포 플롯
        self.plot_prediction_distribution(report['distribution_analysis'], plots_dir)

        # 5. 확률 분포 히스토그램
        self.plot_probability_distributions(predictions, plots_dir)

    def plot_threshold_sensitivity(self, threshold_analysis: Dict[str, List[float]],
                                 output_dir: str):
        """임계값 민감도 플롯"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Threshold Sensitivity Analysis', fontsize=16)

        thresholds = threshold_analysis['thresholds']

        # F1 Score
        axes[0, 0].plot(thresholds, threshold_analysis['macro_f1'], 'bo-')
        axes[0, 0].set_title('Macro F1 Score')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].grid(True)

        # Exact Match Accuracy
        axes[0, 1].plot(thresholds, threshold_analysis['exact_match_accuracy'], 'ro-')
        axes[0, 1].set_title('Exact Match Accuracy')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)

        # Precision & Recall
        axes[1, 0].plot(thresholds, threshold_analysis['macro_precision'], 'go-', label='Precision')
        axes[1, 0].plot(thresholds, threshold_analysis['macro_recall'], 'mo-', label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Hamming Loss
        axes[1, 1].plot(thresholds, threshold_analysis['hamming_loss'], 'co-')
        axes[1, 1].set_title('Hamming Loss')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Hamming Loss')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_per_label_performance(self, per_label_metrics: Dict[str, Dict[str, float]],
                                 output_dir: str):
        """레이블별 성능 플롯"""

        VisualizationUtils.plot_per_label_metrics(
            per_label_metrics,
            os.path.join(output_dir, 'per_label_performance.png')
        )

    def plot_prediction_distribution(self, distribution_analysis: Dict[str, Any],
                                   output_dir: str):
        """예측 분포 플롯"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 레이블별 예측 빈도
        labels = list(distribution_analysis['label_prediction_counts'].keys())
        counts = list(distribution_analysis['label_prediction_counts'].values())

        bars = axes[0].bar(range(len(labels)), counts)
        axes[0].set_title('Label Prediction Frequency')
        axes[0].set_xlabel('Labels')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=45, ha='right')

        # 막대 위에 수치 표시
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(counts),
                        str(count), ha='center', va='bottom', fontsize=9)

        # 상위 레이블 조합
        top_combinations = distribution_analysis['top_label_combinations'][:5]
        combination_labels = [', '.join(combo['labels']) for combo in top_combinations]
        combination_counts = [combo['count'] for combo in top_combinations]

        bars = axes[1].bar(range(len(combination_labels)), combination_counts)
        axes[1].set_title('Top Label Combinations')
        axes[1].set_xlabel('Label Combinations')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(len(combination_labels)))
        axes[1].set_xticklabels(combination_labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_probability_distributions(self, predictions: np.ndarray, output_dir: str):
        """확률 분포 히스토그램"""

        n_labels = len(self.label_columns)
        n_cols = 4
        n_rows = (n_labels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        fig.suptitle('Probability Distributions per Label', fontsize=16)

        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        for i, label in enumerate(self.label_columns):
            axes[i].hist(predictions[:, i], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{label}')
            axes[i].set_xlabel('Probability')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
            axes[i].legend()

        # 빈 서브플롯 숨기기
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="훈련된 모델 경로")
    parser.add_argument("--config_path", type=str, help="설정 파일 경로")
    parser.add_argument("--data_file", type=str, required=True, help="평가할 데이터 파일")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="결과 저장 디렉토리")
    parser.add_argument("--threshold", type=float, default=0.5, help="분류 임계값")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--device", type=str, help="사용할 디바이스 (cuda/cpu)")

    args = parser.parse_args()

    # 평가기 생성
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )

    # 평가 수행
    try:
        report = evaluator.create_evaluation_report(
            data_file=args.data_file,
            output_dir=args.output_dir,
            threshold=args.threshold,
            batch_size=args.batch_size
        )

        # 결과 요약 출력
        print("\n=== 평가 결과 요약 ===")
        print(f"데이터 파일: {args.data_file}")
        print(f"총 샘플 수: {report['dataset_info']['total_samples']:,}")
        print(f"임계값: {args.threshold}")
        print(f"최적 임계값: {report['model_info']['best_threshold']:.3f}")

        overall = report['overall_metrics']
        print(f"\n전체 성능:")
        print(f"  Exact Match Accuracy: {overall['exact_match_accuracy']:.4f}")
        print(f"  Macro F1: {overall['macro_f1']:.4f}")
        print(f"  Macro Precision: {overall['macro_precision']:.4f}")
        print(f"  Macro Recall: {overall['macro_recall']:.4f}")
        print(f"  Hamming Loss: {overall['hamming_loss']:.4f}")

        if 'macro_auc' in overall and overall['macro_auc'] is not None:
            print(f"  Macro AUC: {overall['macro_auc']:.4f}")

        print(f"\n결과가 {args.output_dir}에 저장되었습니다.")

    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()