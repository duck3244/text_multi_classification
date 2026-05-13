
"""
data_loader.py
Korean UnSmile 데이터 로딩 유틸리티
"""

import pandas as pd
import json
import ast
from typing import Dict, Tuple, List


def load_korean_unsmile_data(data_dir: str = 'korean_unsmile_csv') -> Dict:
    """Korean UnSmile 데이터셋 로드"""
    
    # 기본 데이터 로드
    train_df = pd.read_csv(f'{data_dir}/korean_unsmile_train.csv')
    valid_df = pd.read_csv(f'{data_dir}/korean_unsmile_valid.csv')
    
    # 레이블 정보 로드
    with open(f'{data_dir}/label_info.json', 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    
    # 클래스 가중치 로드
    with open(f'{data_dir}/class_weights.json', 'r', encoding='utf-8') as f:
        class_weights = json.load(f)
    
    # 데이터 통계 로드
    with open(f'{data_dir}/dataset_statistics.json', 'r', encoding='utf-8') as f:
        statistics = json.load(f)
    
    print(f"데이터 로드 완료!")
    print(f"학습 데이터: {len(train_df):,}개")
    print(f"검증 데이터: {len(valid_df):,}개")
    print(f"레이블 수: {label_info['num_labels']}개")
    
    return {
        'train_df': train_df,
        'valid_df': valid_df,
        'label_info': label_info,
        'class_weights': class_weights,
        'statistics': statistics
    }


def load_with_labels_list(data_dir: str = 'korean_unsmile_csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """레이블이 리스트 형태로 된 데이터 로드"""
    
    train_df = pd.read_csv(f'{data_dir}/korean_unsmile_train_with_labels.csv')
    valid_df = pd.read_csv(f'{data_dir}/korean_unsmile_valid_with_labels.csv')
    
    # 문자열로 저장된 리스트를 실제 리스트로 변환
    train_df['labels'] = train_df['labels'].apply(ast.literal_eval)
    valid_df['labels'] = valid_df['labels'].apply(ast.literal_eval)
    
    return train_df, valid_df


def get_label_info(data_dir: str = 'korean_unsmile_csv') -> Dict:
    """레이블 정보만 로드"""
    
    with open(f'{data_dir}/label_info.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def get_class_weights(data_dir: str = 'korean_unsmile_csv') -> Dict:
    """클래스 가중치만 로드"""
    
    with open(f'{data_dir}/class_weights.json', 'r', encoding='utf-8') as f:
        return json.load(f)


# 사용 예제
if __name__ == "__main__":
    # 기본 데이터 로드
    data = load_korean_unsmile_data()
    
    # 레이블 리스트 형태 데이터 로드
    train_df, valid_df = load_with_labels_list()
    
    print("\n샘플 데이터:")
    print(train_df.head())
