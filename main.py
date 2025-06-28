#!/usr/bin/env python3
"""
main.py
Korean UnSmile 다중 레이블 분류 프로젝트 메인 실행 스크립트
"""

import os
import sys
import argparse

from typing import Optional


def setup_project():
    """프로젝트 초기 설정"""
    print("Korean UnSmile Multi-label Classification Project")
    print("=" * 50)

    # 필요한 라이브러리 확인
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),  # import명과 설치명이 다름
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm')
    ]

    missing_packages = []
    for import_name, install_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(install_name)

    if missing_packages:
        print(f"다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("모든 필수 패키지가 설치되어 있습니다!")
    return True


def process_data(args):
    """데이터 처리"""
    print("\n=== 데이터 처리 단계 ===")

    from data_processor import KoreanUnsmileProcessor

    processor = KoreanUnsmileProcessor(output_dir=args.data_output_dir)

    try:
        all_df, train_df, valid_df = processor.process()
        print(f"데이터 처리 완료! 결과: {args.data_output_dir}")
        return True
    except Exception as e:
        print(f"데이터 처리 중 오류: {e}")
        return False


def train_model(args):
    """모델 훈련"""
    print("\n=== 모델 훈련 단계 ===")

    from trainer import KoreanUnsmileTrainer
    from config import create_config_from_args, validate_config

    try:
        # 설정 생성
        if args.config and os.path.exists(args.config):
            from config import ExperimentConfig
            config = ExperimentConfig.load(args.config)
        else:
            config = create_config_from_args(args)

        # 설정 검증
        validate_config(config)

        # 훈련 시작
        trainer = KoreanUnsmileTrainer(config)
        final_metrics = trainer.train()

        print(f"훈련 완료! 결과: {config.logging.output_dir}")

        # 주요 결과 출력
        print(f"\n주요 성능 지표:")
        print(f"  Exact Match Accuracy: {final_metrics['overall']['exact_match_accuracy']:.4f}")
        print(f"  Macro F1: {final_metrics['overall']['macro_f1']:.4f}")
        print(f"  Hamming Loss: {final_metrics['overall']['hamming_loss']:.4f}")

        return config.logging.output_dir

    except Exception as e:
        print(f"훈련 중 오류: {e}")
        return None


def evaluate_model(args):
    """모델 평가"""
    print("\n=== 모델 평가 단계 ===")

    from evaluator import ModelEvaluator

    # 모델 경로 확인
    if not os.path.exists(args.model_path):
        print(f"모델 파일을 찾을 수 없습니다: {args.model_path}")
        return False

    # 평가 데이터 확인
    if not os.path.exists(args.eval_data):
        print(f"평가 데이터를 찾을 수 없습니다: {args.eval_data}")
        return False

    try:
        # 평가기 생성
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            config_path=args.config,
            device=args.device
        )

        # 평가 수행
        report = evaluator.create_evaluation_report(
            data_file=args.eval_data,
            output_dir=args.eval_output_dir,
            threshold=args.threshold,
            batch_size=args.batch_size
        )

        print(f"평가 완료! 결과: {args.eval_output_dir}")

        # 주요 결과 출력
        overall = report['overall_metrics']
        print(f"\n평가 결과:")
        print(f"  샘플 수: {report['dataset_info']['total_samples']:,}")
        print(f"  Exact Match Accuracy: {overall['exact_match_accuracy']:.4f}")
        print(f"  Macro F1: {overall['macro_f1']:.4f}")
        print(f"  Macro Precision: {overall['macro_precision']:.4f}")
        print(f"  Macro Recall: {overall['macro_recall']:.4f}")
        print(f"  Hamming Loss: {overall['hamming_loss']:.4f}")

        return True

    except Exception as e:
        print(f"평가 중 오류: {e}")
        return False


def pipeline_mode(args):
    """파이프라인 모드 (데이터 처리 -> 훈련 -> 평가)"""
    print("\n=== 전체 파이프라인 실행 ===")

    # 1. 데이터 처리
    if not process_data(args):
        print("데이터 처리 실패")
        return False

    # 2. 모델 훈련
    training_output_dir = train_model(args)
    if not training_output_dir:
        print("모델 훈련 실패")
        return False

    # 3. 모델 평가 (검증 데이터로)
    # 훈련에서 생성된 베스트 모델 사용
    best_model_path = os.path.join(training_output_dir, 'checkpoints', 'best_model.pth')

    if os.path.exists(best_model_path):
        # 평가 인자 업데이트
        args.model_path = best_model_path
        args.eval_data = os.path.join(args.data_output_dir, 'korean_unsmile_valid.csv')
        args.eval_output_dir = os.path.join(training_output_dir, 'evaluation')

        if not evaluate_model(args):
            print("모델 평가 실패")
            return False
    else:
        print("베스트 모델을 찾을 수 없습니다.")
        return False

    print("\n=== 전체 파이프라인 완료 ===")
    print(f"훈련 결과: {training_output_dir}")
    print(f"평가 결과: {args.eval_output_dir}")

    return True


def create_parser():
    """명령행 인자 파서 생성"""

    parser = argparse.ArgumentParser(
        description="Korean UnSmile Multi-label Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행
  python main.py pipeline

  # 데이터만 처리
  python main.py process-data

  # 모델만 훈련
  python main.py train --data_dir korean_unsmile_csv

  # 모델만 평가
  python main.py evaluate --model_path output/checkpoints/best_model.pth --eval_data data.csv
        """
    )

    # 서브커맨드 생성
    subparsers = parser.add_subparsers(dest='command', help='실행할 작업')

    # 파이프라인 모드
    pipeline_parser = subparsers.add_parser('pipeline', help='전체 파이프라인 실행')
    add_common_args(pipeline_parser)
    add_training_args(pipeline_parser)

    # 데이터 처리
    data_parser = subparsers.add_parser('process-data', help='데이터 처리만 실행')
    data_parser.add_argument('--data_output_dir', type=str, default='korean_unsmile_csv',
                             help='처리된 데이터 저장 디렉토리')

    # 모델 훈련
    train_parser = subparsers.add_parser('train', help='모델 훈련만 실행')
    add_common_args(train_parser)
    add_training_args(train_parser)

    # 모델 평가
    eval_parser = subparsers.add_parser('evaluate', help='모델 평가만 실행')
    add_evaluation_args(eval_parser)

    # 설정 생성
    config_parser = subparsers.add_parser('create-config', help='설정 파일 생성')
    add_config_args(config_parser)

    return parser


def add_common_args(parser):
    """공통 인자 추가"""
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--data_output_dir', type=str, default='korean_unsmile_csv',
                        help='데이터 디렉토리')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--device', type=str, help='사용할 디바이스 (cuda/cpu)')


def add_training_args(parser):
    """훈련 관련 인자 추가"""

    # 모델 관련
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument('--model_name', type=str,
                             default='monologg/koelectra-base-v3-discriminator',
                             help='사전 훈련된 모델명')
    model_group.add_argument('--max_length', type=int, default=512,
                             help='최대 시퀀스 길이')
    model_group.add_argument('--dropout_rate', type=float, default=0.1,
                             help='드롭아웃 비율')

    # 훈련 관련
    training_group = parser.add_argument_group('Training Arguments')
    training_group.add_argument('--batch_size', type=int, default=16,
                                help='배치 크기')
    training_group.add_argument('--learning_rate', type=float, default=2e-5,
                                help='학습률')
    training_group.add_argument('--num_epochs', type=int, default=5,
                                help='훈련 에포크 수')
    training_group.add_argument('--weight_decay', type=float, default=0.01,
                                help='가중치 감쇠')
    training_group.add_argument('--warmup_ratio', type=float, default=0.1,
                                help='웜업 비율')
    training_group.add_argument('--eval_steps', type=int, default=500,
                                help='평가 주기 (스텝)')
    training_group.add_argument('--save_steps', type=int, default=1000,
                                help='저장 주기 (스텝)')
    training_group.add_argument('--early_stopping_patience', type=int, default=3,
                                help='조기 종료 patience')

    # 데이터 관련
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument('--data_dir', type=str, default='korean_unsmile_csv',
                            help='데이터 디렉토리')
    data_group.add_argument('--max_samples', type=int,
                            help='최대 샘플 수 (디버깅용)')

    # 로깅 관련
    logging_group = parser.add_argument_group('Logging Arguments')
    logging_group.add_argument('--output_dir', type=str, default='output',
                               help='출력 디렉토리')
    logging_group.add_argument('--run_name', type=str,
                               help='실행 이름')
    logging_group.add_argument('--use_wandb', action='store_true',
                               help='WandB 사용')
    logging_group.add_argument('--wandb_project', type=str,
                               default='korean-unsmile-classification',
                               help='WandB 프로젝트명')

    # 프리셋
    parser.add_argument('--preset', type=str, default='default',
                        choices=['default', 'debug', 'large_batch', 'high_lr', 'conservative'],
                        help='사전 정의된 설정')


def add_evaluation_args(parser):
    """평가 관련 인자 추가"""
    parser.add_argument('--model_path', type=str, required=True,
                        help='평가할 모델 경로')
    parser.add_argument('--config', type=str,
                        help='모델 설정 파일 경로')
    parser.add_argument('--eval_data', type=str, required=True,
                        help='평가 데이터 파일')
    parser.add_argument('--eval_output_dir', type=str, default='evaluation_results',
                        help='평가 결과 저장 디렉토리')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='분류 임계값')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--device', type=str,
                        help='사용할 디바이스')


def add_config_args(parser):
    """설정 생성 관련 인자 추가"""
    parser.add_argument('--preset', type=str, default='default',
                        choices=['default', 'debug', 'large_batch', 'high_lr', 'conservative'],
                        help='사전 정의된 설정')
    parser.add_argument('--output_file', type=str, default='experiment_config.json',
                        help='설정 파일 저장 경로')
    parser.add_argument('--experiment_name', type=str, default='koelectra_multilabel',
                        help='실험 이름')


def create_config_command(args):
    """설정 파일 생성"""
    print("\n=== 설정 파일 생성 ===")

    from config import ConfigManager

    try:
        config_manager = ConfigManager()
        config = config_manager.create_config(args.preset)

        # 실험 이름 설정
        config.experiment_name = args.experiment_name

        # 설정 저장
        config.save(args.output_file)

        print(f"설정 파일 생성 완료: {args.output_file}")
        print(f"프리셋: {args.preset}")
        print(f"실험 이름: {args.experiment_name}")

        # 주요 설정 출력
        print(f"\n주요 설정:")
        print(f"  모델: {config.model.model_name}")
        print(f"  배치 크기: {config.training.batch_size}")
        print(f"  학습률: {config.training.learning_rate}")
        print(f"  에포크: {config.training.num_epochs}")

        return True

    except Exception as e:
        print(f"설정 파일 생성 중 오류: {e}")
        return False


def print_system_info():
    """시스템 정보 출력"""
    import torch

    print("\n=== 시스템 정보 ===")
    print(f"Python 버전: {sys.version}")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def main():
    """메인 함수"""

    # 시스템 정보 출력
    print_system_info()

    # 프로젝트 설정 확인
    if not setup_project():
        sys.exit(1)

    # 명령행 인자 파싱
    parser = create_parser()
    args = parser.parse_args()

    # 명령어 없이 실행된 경우
    if not args.command:
        parser.print_help()
        print(f"\n힌트: 전체 파이프라인을 실행하려면 'python {sys.argv[0]} pipeline'을 사용하세요.")
        sys.exit(1)

    # 명령어별 실행
    success = False

    try:
        if args.command == 'pipeline':
            success = pipeline_mode(args)

        elif args.command == 'process-data':
            success = process_data(args)

        elif args.command == 'train':
            result = train_model(args)
            success = result is not None

        elif args.command == 'evaluate':
            success = evaluate_model(args)

        elif args.command == 'create-config':
            success = create_config_command(args)

        else:
            print(f"알 수 없는 명령어: {args.command}")
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        sys.exit(1)

    except Exception as e:
        print(f"\n예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 결과 확인
    if success:
        print("\n✅ 작업이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 작업이 실패했습니다.")
        sys.exit(1)


def quick_start():
    """빠른 시작 가이드"""
    print("Korean UnSmile Multi-label Classification - 빠른 시작 가이드")
    print("=" * 60)
    print()

    print("1. 필요한 패키지 설치:")
    print("   pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn tqdm")
    print()

    print("2. 전체 파이프라인 실행:")
    print("   python main.py pipeline")
    print()

    print("3. 개별 단계 실행:")
    print("   # 데이터 처리")
    print("   python main.py process-data")
    print()
    print("   # 모델 훈련")
    print("   python main.py train")
    print()
    print("   # 모델 평가")
    print(
        "   python main.py evaluate --model_path output/checkpoints/best_model.pth --eval_data korean_unsmile_csv/korean_unsmile_valid.csv")
    print()

    print("4. 고급 옵션:")
    print("   # 디버그 모드 (빠른 테스트)")
    print("   python main.py train --preset debug")
    print()
    print("   # 커스텀 설정")
    print("   python main.py train --batch_size 8 --learning_rate 1e-5 --num_epochs 3")
    print()
    print("   # 설정 파일 생성")
    print("   python main.py create-config --preset conservative --output_file my_config.json")
    print("   python main.py train --config my_config.json")
    print()

    print("더 자세한 옵션은 'python main.py --help' 또는 'python main.py [command] --help'를 확인하세요.")


if __name__ == "__main__":
    # 인자 없이 실행되면 빠른 시작 가이드 표시
    if len(sys.argv) == 1:
        quick_start()
        print("\n계속하려면 명령어를 입력하세요. 예: python main.py pipeline")
        sys.exit(0)

    main()