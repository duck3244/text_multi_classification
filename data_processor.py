"""
data_processor.py
Korean UnSmile 데이터셋 다운로드 및 전처리
"""

import os
import ast
import json
import numpy as np
import pandas as pd

from datasets import load_dataset
from typing import Dict, List, Tuple, Optional


class KoreanUnsmileProcessor:
    """Korean UnSmile 데이터셋 처리 클래스"""
    
    def __init__(self, output_dir: str = 'korean_unsmile_csv'):
        self.output_dir = output_dir
        self.label_columns = [
            '여성/가족', '남성', '성소수자', '인종/국적', '연령',
            '지역', '종교', '기타 혐오', '악플/욕설', 'clean'
        ]
        self.label_descriptions = {
            '여성/가족': '여성성 및 여성의 성역할에 대한 통념, 여성 차별 희화화, 페미니즘 관련 악플',
            '남성': '집단으로서의 남성 일반을 비하, 조롱, 희화화하는 발언',
            '성소수자': '성소수자를 배척하거나 희화화하는 표현',
            '인종/국적': '특정 인종과 국적에 대한 욕설, 고정관념, 조롱',
            '연령': '특정 세대나 연령을 비하하는 은어 및 혐오 표현',
            '지역': '특정 지역에 대한 은어 및 혐오 표현',
            '종교': '특정 종교에 대한 혐오 및 종교인 집단에 대한 비난',
            '기타 혐오': '위 카테고리 이외의 집단을 대상으로 하는 혐오 표현',
            '악플/욕설': '집단 지칭 없는 비하/욕설, 불쾌감 주는 내용',
            'clean': '혐오표현, 욕설, 불쾌감, 음란성 내용이 없는 일반 문장'
        }
    
    def download_dataset(self) -> Dict:
        """Hugging Face에서 데이터셋 다운로드"""
        print("Korean UnSmile 데이터셋 다운로드 중...")
        
        try:
            dataset = load_dataset('smilegate-ai/kor_unsmile')
            print("데이터셋 로드 완료!")
            return dataset
        except Exception as e:
            print(f"데이터셋 로드 실패: {e}")
            raise
    
    def process_split(self, data, split_name: str) -> pd.DataFrame:
        """데이터 분할을 처리하여 DataFrame으로 변환"""
        
        texts = []
        labels_list = []
        
        for example in data:
            text = example['문장']
            
            # 각 레이블에 대한 값 추출 (0 또는 1)
            labels = []
            for label_col in self.label_columns:
                if label_col in example:
                    labels.append(example[label_col])
                elif label_col == '기타 혐오' and '기타혐오' in example:
                    labels.append(example['기타혐오'])
                else:
                    labels.append(0)  # 기본값
            
            texts.append(text)
            labels_list.append(labels)
        
        # DataFrame 생성
        df = pd.DataFrame({'text': texts})
        
        # 각 레이블 컬럼 추가
        for i, label_col in enumerate(self.label_columns):
            df[label_col] = [labels[i] for labels in labels_list]
        
        return df
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """데이터 통계 계산"""
        
        # 레이블별 분포
        label_distribution = {}
        for label_col in self.label_columns:
            count = df[label_col].sum()
            percentage = count / len(df) * 100
            label_distribution[label_col] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # 다중 레이블 통계
        label_counts_per_sample = df[self.label_columns].sum(axis=1)
        
        statistics = {
            'total_samples': len(df),
            'label_distribution': label_distribution,
            'avg_labels_per_sample': float(label_counts_per_sample.mean()),
            'max_labels_per_sample': int(label_counts_per_sample.max()),
            'samples_with_no_labels': int((label_counts_per_sample == 0).sum()),
            'samples_with_multiple_labels': int((label_counts_per_sample > 1).sum())
        }
        
        return statistics
    
    def calculate_class_weights(self, df: pd.DataFrame) -> Dict:
        """클래스 불균형 처리를 위한 가중치 계산"""
        
        class_weights = {}
        total_samples = len(df)
        
        for label_col in self.label_columns:
            pos_count = df[label_col].sum()
            neg_count = total_samples - pos_count
            
            if pos_count > 0:
                weight = neg_count / pos_count
                class_weights[label_col] = round(weight, 3)
        
        return class_weights
    
    def create_labels_list_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """레이블을 리스트 형태로 변환한 DataFrame 생성"""
        
        df_copy = df.copy()
        df_copy['labels'] = df_copy[self.label_columns].values.tolist()
        return df_copy[['text', 'labels']].copy()
    
    def save_data_files(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                       all_df: pd.DataFrame) -> None:
        """데이터 파일들 저장"""
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 기본 CSV 파일들
        all_df.to_csv(f'{self.output_dir}/korean_unsmile_all.csv', index=False, encoding='utf-8')
        train_df.to_csv(f'{self.output_dir}/korean_unsmile_train.csv', index=False, encoding='utf-8')
        valid_df.to_csv(f'{self.output_dir}/korean_unsmile_valid.csv', index=False, encoding='utf-8')
        
        print(f"기본 CSV 파일 저장 완료!")
        
        # 2. 레이블 리스트 형태 CSV 파일들
        all_labels_df = self.create_labels_list_format(all_df)
        train_labels_df = self.create_labels_list_format(train_df)
        valid_labels_df = self.create_labels_list_format(valid_df)
        
        all_labels_df.to_csv(f'{self.output_dir}/korean_unsmile_all_with_labels.csv', index=False, encoding='utf-8')
        train_labels_df.to_csv(f'{self.output_dir}/korean_unsmile_train_with_labels.csv', index=False, encoding='utf-8')
        valid_labels_df.to_csv(f'{self.output_dir}/korean_unsmile_valid_with_labels.csv', index=False, encoding='utf-8')
        
        print(f"레이블 리스트 형태 CSV 파일 저장 완료!")
    
    def save_metadata(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                     all_df: pd.DataFrame) -> None:
        """메타데이터 파일들 저장"""
        
        # 1. 클래스 가중치
        class_weights = self.calculate_class_weights(all_df)
        with open(f'{self.output_dir}/class_weights.json', 'w', encoding='utf-8') as f:
            json.dump(class_weights, f, ensure_ascii=False, indent=2)
        
        # 2. 레이블 정보
        label_info = {
            'label_columns': self.label_columns,
            'num_labels': len(self.label_columns),
            'label_descriptions': self.label_descriptions
        }
        with open(f'{self.output_dir}/label_info.json', 'w', encoding='utf-8') as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        
        # 3. 데이터 통계
        statistics = {
            'train': self.calculate_statistics(train_df),
            'valid': self.calculate_statistics(valid_df),
            'all': self.calculate_statistics(all_df)
        }
        with open(f'{self.output_dir}/dataset_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        print(f"메타데이터 파일 저장 완료!")
    
    def print_statistics(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                        all_df: pd.DataFrame) -> None:
        """데이터 통계 출력"""
        
        print("\n=== 데이터셋 정보 ===")
        print(f"전체 데이터 수: {len(all_df):,}")
        print(f"학습 데이터 수: {len(train_df):,}")
        print(f"검증 데이터 수: {len(valid_df):,}")
        
        print("\n=== 레이블별 분포 ===")
        for label_col in self.label_columns:
            count = all_df[label_col].sum()
            percentage = count / len(all_df) * 100
            print(f"{label_col}: {count:,}개 ({percentage:.1f}%)")
        
        # 다중 레이블 통계
        label_counts_per_sample = all_df[self.label_columns].sum(axis=1)
        print("\n=== 다중 레이블 통계 ===")
        print(f"평균 레이블 수: {label_counts_per_sample.mean():.2f}")
        print(f"최대 레이블 수: {label_counts_per_sample.max()}")
        print(f"레이블이 없는 샘플: {(label_counts_per_sample == 0).sum():,}개")
        print(f"다중 레이블 샘플: {(label_counts_per_sample > 1).sum():,}개")
        
        # 샘플 데이터 출력
        print("\n=== 샘플 데이터 ===")
        print(all_df.head())
    
    def create_data_loader_script(self) -> None:
        """데이터 로딩을 위한 스크립트 생성"""
        
        script_content = '''
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
    
    print("\\n샘플 데이터:")
    print(train_df.head())
'''
        
        with open(f'{self.output_dir}/data_loader.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"데이터 로더 스크립트 저장: {self.output_dir}/data_loader.py")
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """전체 데이터 처리 프로세스 실행"""
        
        # 데이터셋 다운로드
        dataset = self.download_dataset()
        
        # 학습용과 검증용 데이터 처리
        train_data = dataset['train']
        valid_data = dataset['valid']
        
        print(f"Train 데이터: {len(train_data):,}개")
        print(f"Validation 데이터: {len(valid_data):,}개")
        
        # 데이터 처리
        print("학습용 데이터 처리 중...")
        train_df = self.process_split(train_data, 'train')
        
        print("검증용 데이터 처리 중...")
        valid_df = self.process_split(valid_data, 'valid')
        
        # 전체 데이터 결합
        print("전체 데이터 결합 중...")
        all_df = pd.concat([train_df, valid_df], ignore_index=True)
        
        # 통계 출력
        self.print_statistics(train_df, valid_df, all_df)
        
        # 데이터 파일 저장
        self.save_data_files(train_df, valid_df, all_df)
        
        # 메타데이터 저장
        self.save_metadata(train_df, valid_df, all_df)
        
        # 데이터 로더 스크립트 생성
        self.create_data_loader_script()
        
        print("\n" + "=" * 60)
        print("데이터 처리 완료!")
        print("생성된 파일들:")
        print(f"- {self.output_dir}/korean_unsmile_all.csv")
        print(f"- {self.output_dir}/korean_unsmile_train.csv")
        print(f"- {self.output_dir}/korean_unsmile_valid.csv")
        print(f"- {self.output_dir}/korean_unsmile_*_with_labels.csv")
        print(f"- {self.output_dir}/class_weights.json")
        print(f"- {self.output_dir}/label_info.json")
        print(f"- {self.output_dir}/dataset_statistics.json")
        print(f"- {self.output_dir}/data_loader.py")
        print("=" * 60)
        
        return all_df, train_df, valid_df


def main():
    """메인 실행 함수"""
    
    print("필요한 라이브러리:")
    print("pip install datasets pandas transformers torch")
    print()
    
    # 데이터 처리기 생성 및 실행
    processor = KoreanUnsmileProcessor()
    all_df, train_df, valid_df = processor.process()
    
    return all_df, train_df, valid_df


if __name__ == "__main__":
    main()