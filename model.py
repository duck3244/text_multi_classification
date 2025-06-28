"""
model.py
KoELECTRA 기반 다중 레이블 분류 모델 정의
"""

import torch
import numpy as np
import torch.nn as nn

from transformers import (
    ElectraTokenizer, 
    ElectraForSequenceClassification,
    ElectraModel,
    ElectraConfig
)
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class KoreanUnsmileDataset(Dataset):
    """Korean UnSmile 데이터셋을 위한 PyTorch Dataset 클래스"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer: ElectraTokenizer, max_length: int = 512):
        """
        Args:
            texts: 입력 텍스트 리스트
            labels: 다중 레이블 리스트 (각 샘플마다 레이블 개수만큼의 0/1 값)
            tokenizer: ELECTRA 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # 텍스트 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }


class MultiLabelElectraClassifier(nn.Module):
    """다중 레이블 분류를 위한 ELECTRA 모델"""
    
    def __init__(self, model_name: str, num_labels: int, 
                 dropout_rate: float = 0.1, hidden_size: Optional[int] = None):
        """
        Args:
            model_name: 사전 훈련된 ELECTRA 모델명
            num_labels: 레이블 개수
            dropout_rate: 드롭아웃 비율
            hidden_size: 은닉층 크기 (None이면 기본값 사용)
        """
        super().__init__()
        
        # ELECTRA 모델 로드
        self.electra = ElectraModel.from_pretrained(model_name)
        
        # 설정 정보
        self.config = self.electra.config
        self.num_labels = num_labels
        
        # 분류기 레이어
        hidden_size = hidden_size or self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """분류기 가중치 초기화"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            labels: 레이블 텐서 [batch_size, num_labels] (선택적)
        
        Returns:
            출력 딕셔너리 (loss, logits)
        """
        
        # ELECTRA 인코딩
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS] 토큰의 출력 사용
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # 분류
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        loss = None
        if labels is not None:
            # 다중 레이블 분류를 위한 BCEWithLogitsLoss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_output
        }
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """예측 수행
        
        Args:
            input_ids: 토큰 ID 텐서
            attention_mask: 어텐션 마스크
            threshold: 이진 분류 임계값
            
        Returns:
            확률값과 이진 예측값
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs['logits'])
            predictions = (probabilities > threshold).float()
            
        return probabilities, predictions
    
    def freeze_electra(self):
        """ELECTRA 부분 동결 (파인튜닝 시 사용)"""
        for param in self.electra.parameters():
            param.requires_grad = False
    
    def unfreeze_electra(self):
        """ELECTRA 부분 동결 해제"""
        for param in self.electra.parameters():
            param.requires_grad = True
    
    def get_input_embeddings(self):
        """입력 임베딩 반환"""
        return self.electra.embeddings.word_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """입력 임베딩 설정"""
        self.electra.embeddings.word_embeddings = new_embeddings


class WeightedBCEWithLogitsLoss(nn.Module):
    """클래스 가중치가 적용된 BCE Loss"""
    
    def __init__(self, pos_weights: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean'):
        """
        Args:
            pos_weights: 각 레이블별 양성 클래스 가중치
            reduction: 손실 축소 방법 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pos_weights = pos_weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 로짓 [batch_size, num_labels]
            labels: 실제 레이블 [batch_size, num_labels]
        """
        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weights,
            reduction=self.reduction
        )
        return loss_fct(logits, labels)


class FocalLoss(nn.Module):
    """다중 레이블을 위한 Focal Loss"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Args:
            alpha: 가중치 인자
            gamma: 포커싱 인자
            reduction: 손실 축소 방법
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 로짓
            labels: 실제 레이블
        """
        # BCE loss 계산
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        
        # 확률 계산
        probs = torch.sigmoid(logits)
        
        # Focal weight 계산
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ModelConfig:
    """모델 설정 클래스"""
    
    def __init__(self):
        # 모델 관련
        self.model_name = "monologg/koelectra-base-v3-discriminator"
        self.num_labels = 10
        self.max_length = 512
        self.dropout_rate = 0.1
        
        # 손실 함수 관련
        self.loss_type = "weighted_bce"  # "bce", "weighted_bce", "focal"
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0
        
        # 기타
        self.hidden_size = None  # None이면 기본값 사용
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """딕셔너리에서 생성"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def create_model(config: ModelConfig, class_weights: Optional[Dict] = None) -> Tuple[MultiLabelElectraClassifier, nn.Module]:
    """모델과 손실 함수 생성
    
    Args:
        config: 모델 설정
        class_weights: 클래스 가중치 딕셔너리
        
    Returns:
        모델과 손실 함수
    """
    
    # 모델 생성
    model = MultiLabelElectraClassifier(
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout_rate=config.dropout_rate,
        hidden_size=config.hidden_size
    )
    
    # 손실 함수 생성
    if config.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    
    elif config.loss_type == "weighted_bce":
        pos_weights = None
        if class_weights:
            # 클래스 가중치를 텐서로 변환
            weights_list = [class_weights.get(f"label_{i}", 1.0) for i in range(config.num_labels)]
            pos_weights = torch.FloatTensor(weights_list)
        
        criterion = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
    
    elif config.loss_type == "focal":
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    
    else:
        raise ValueError(f"지원하지 않는 손실 함수 타입: {config.loss_type}")
    
    return model, criterion


def load_tokenizer(model_name: str) -> ElectraTokenizer:
    """토크나이저 로드"""
    return ElectraTokenizer.from_pretrained(model_name)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """모델 파라미터 개수 계산
    
    Returns:
        (전체 파라미터 수, 훈련 가능한 파라미터 수)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def model_summary(model: MultiLabelElectraClassifier) -> Dict:
    """모델 요약 정보 반환"""
    
    total_params, trainable_params = count_parameters(model)
    
    summary = {
        'model_name': model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'unknown',
        'num_labels': model.num_labels,
        'hidden_size': model.config.hidden_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'vocab_size': model.config.vocab_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'num_attention_heads': model.config.num_attention_heads,
        'num_hidden_layers': model.config.num_hidden_layers
    }
    
    return summary


# 사용 예제
if __name__ == "__main__":
    # 설정 생성
    config = ModelConfig()
    config.num_labels = 10
    
    # 모델 생성
    model, criterion = create_model(config)
    
    # 모델 요약
    summary = model_summary(model)
    print("모델 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # 토크나이저 로드
    tokenizer = load_tokenizer(config.model_name)
    
    # 테스트 데이터
    texts = ["안녕하세요", "좋은 하루입니다"]
    labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    
    # 데이터셋 생성
    dataset = KoreanUnsmileDataset(texts, labels, tokenizer)
    
    print(f"\n데이터셋 크기: {len(dataset)}")
    print(f"샘플 데이터: {dataset[0]['input_ids'].shape}")
    
    # 순전파 테스트
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        input_ids = sample['input_ids'].unsqueeze(0)
        attention_mask = sample['attention_mask'].unsqueeze(0)
        
        outputs = model(input_ids, attention_mask)
        print(f"출력 로짓 형태: {outputs['logits'].shape}")
        print(f"출력 로짓: {outputs['logits']}")
        
        # 예측
        probs, preds = model.predict(input_ids, attention_mask)
        print(f"예측 확률: {probs}")
        print(f"예측 결과: {preds}")
