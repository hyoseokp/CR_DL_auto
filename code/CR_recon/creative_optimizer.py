"""
Creative Optimizer: 훈련 로그를 분석하고 창의적인 모델과 손실 함수를 제안
사용: python creative_optimizer.py --log outputs/train_log.txt
"""
import argparse
import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess


def parse_train_log(log_path: str) -> Dict[str, Any]:
    """훈련 로그를 파싱하여 설정과 결과 추출"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {
        'model_name': None,
        'model_path': None,
        'model_params': {},
        'loss_name': None,
        'loss_path': None,
        'loss_params': {},
        'hyperparams': {},
        'best_val_loss': float('inf'),
        'final_val_loss': float('inf'),
        'epochs_trained': 0,
        'train_losses': [],
        'val_losses': [],
        'raw_content': content,
    }

    # Model 정보 추출
    model_match = re.search(r'Model: (\w+)', content)
    if model_match:
        result['model_name'] = model_match.group(1)

    model_path_match = re.search(r'Model Path: ([\w/._-]+)', content)
    if model_path_match:
        result['model_path'] = model_path_match.group(1)

    # Model Parameters 추출
    model_params_section = re.search(
        r'Model Parameters:(.*?)(?=Loss Function|Hyperparameters)',
        content,
        re.DOTALL
    )
    if model_params_section:
        params_text = model_params_section.group(1)
        for line in params_text.split('\n'):
            match = re.search(r'-\s*(\w+):\s*(.*)', line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2).strip()
                try:
                    result['model_params'][param_name] = yaml.safe_load(param_value)
                except:
                    result['model_params'][param_name] = param_value

    # Loss 정보 추출
    loss_match = re.search(r'Loss Function: (\w+)', content)
    if loss_match:
        result['loss_name'] = loss_match.group(1)

    loss_path_match = re.search(r'Loss Path: ([\w/._-]+)', content)
    if loss_path_match:
        result['loss_path'] = loss_path_match.group(1)

    # Loss Parameters 추출
    loss_params_section = re.search(
        r'Loss Parameters:(.*?)(?=Hyperparameters)',
        content,
        re.DOTALL
    )
    if loss_params_section:
        params_text = loss_params_section.group(1)
        for line in params_text.split('\n'):
            match = re.search(r'-\s*(\w+):\s*(.*)', line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2).strip()
                try:
                    result['loss_params'][param_name] = yaml.safe_load(param_value)
                except:
                    result['loss_params'][param_name] = param_value

    # Hyperparameters 추출
    hyper_section = re.search(
        r'Hyperparameters:(.*?)(?=Data Statistics)',
        content,
        re.DOTALL
    )
    if hyper_section:
        hyper_text = hyper_section.group(1)
        for line in hyper_text.split('\n'):
            match = re.search(r'-\s*([\w\s]+):\s*(.*)', line)
            if match:
                param_name = match.group(1).strip()
                param_value = match.group(2).strip()
                try:
                    result['hyperparams'][param_name] = yaml.safe_load(param_value)
                except:
                    result['hyperparams'][param_name] = param_value

    # Best val loss 추출 (모든 epoch의 손실값 수집)
    best_loss_matches = re.findall(
        r'best_val=([\d.e+-]+)',
        content
    )
    if best_loss_matches:
        result['best_val_loss'] = float(best_loss_matches[-1])
        result['final_val_loss'] = float(best_loss_matches[-1])
        result['val_losses'] = [float(x) for x in best_loss_matches]

    # Train losses 추출
    train_loss_matches = re.findall(
        r'train_loss=([\d.e+-]+)',
        content
    )
    if train_loss_matches:
        result['train_losses'] = [float(x) for x in train_loss_matches]

    # Epochs trained 추출
    epoch_matches = re.findall(
        r'\[EPOCH\]\s+(\d+)/(\d+)',
        content
    )
    if epoch_matches:
        result['epochs_trained'] = int(epoch_matches[-1][0])

    return result


def analyze_performance(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """성능 지표를 분석하여 개선 방향 파악"""
    analysis = {
        'current_best_loss': log_data['best_val_loss'],
        'final_loss': log_data['final_val_loss'],
        'epochs_trained': log_data['epochs_trained'],
        'model_name': log_data['model_name'],
        'loss_name': log_data['loss_name'],
        'metrics': {}
    }

    # 손실 감소 추이 분석
    if len(log_data['val_losses']) > 1:
        recent_losses = log_data['val_losses'][-min(10, len(log_data['val_losses'])):]
        first_loss = log_data['val_losses'][0]
        last_loss = log_data['val_losses'][-1]

        improvement = (first_loss - last_loss) / first_loss if first_loss > 0 else 0
        analysis['metrics']['improvement_rate'] = improvement
        analysis['metrics']['convergence_trend'] = 'improving' if last_loss < recent_losses[0] else 'plateauing'

    # 과적합 여부 확인
    if len(log_data['train_losses']) > 0 and len(log_data['val_losses']) > 0:
        train_mean = sum(log_data['train_losses']) / len(log_data['train_losses'])
        val_mean = sum(log_data['val_losses']) / len(log_data['val_losses'])
        gap = (val_mean - train_mean) / train_mean if train_mean > 0 else 0
        analysis['metrics']['train_val_gap'] = gap
        analysis['metrics']['overfitting'] = 'likely' if gap > 0.2 else 'moderate' if gap > 0.1 else 'unlikely'

    return analysis


def propose_creative_models(log_data: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict]:
    """창의적인 모델 아키텍처 제안"""
    proposals = []
    current_model = log_data['model_name']
    current_loss = log_data['best_val_loss']

    # 제안 1: 만약 CNN_XAttn을 사용 중이면 → Attention-Enhanced CNN
    if current_model == 'cnn_xattn':
        proposals.append({
            'type': 'model',
            'name': 'MultiHeadChannelAttention',
            'description': 'Multi-head channel attention을 추가한 개선된 CNN 아키텍처',
            'rationale': 'Transformer의 Multi-head mechanism을 더 효율적으로 적용하여 다양한 채널 특성 학습',
            'architecture_sketch': {
                'stem': '동일 (5x5 conv, stride=2)',
                'backbone': 'Residual blocks with Multi-Head Channel Attention (4 heads)',
                'decoder': '간단한 feedforward 대신 Residual connections으로 강화',
            },
            'expected_benefit': 'attention 메커니즘 강화로 특성 추출 향상',
            'implementation_file': 'models/multihead_attention_cnn.py',
            'config_changes': {'model': {'name': 'multihead_attention_cnn', 'params': {'heads': 4, 'head_dim': 32}}}
        })

    # 제안 2: 만약 과적합이 심하면 → Regularized Model
    if analysis['metrics'].get('overfitting') == 'likely':
        proposals.append({
            'type': 'model',
            'name': 'RegularizedCNN',
            'description': 'Batch normalization, Layer normalization, Attention dropout을 강화한 모델',
            'rationale': '과적합 증상 → 정규화 메커니즘 강화 필요',
            'architecture_sketch': {
                'normalization': 'GroupNorm → LayerNorm + GroupNorm 혼합',
                'dropout': 'Spatial dropout 강화',
                'attention': 'Attention weight 정규화 추가',
            },
            'expected_benefit': '과적합 감소 및 일반화 성능 향상',
            'implementation_file': 'models/regularized_cnn.py',
            'config_changes': {
                'model': {
                    'name': 'regularized_cnn',
                    'params': {
                        'cnn_dropout': 0.1,
                        'attention_dropout': 0.15,
                        'use_layer_norm': True
                    }
                }
            }
        })

    # 제안 3: 수렴 정체되었다면 → Residual Path 강화
    if analysis['metrics'].get('convergence_trend') == 'plateauing' and current_loss > 0.01:
        proposals.append({
            'type': 'model',
            'name': 'DenseResidualCNN',
            'description': 'DenseNet 스타일의 Dense residual connections 추가',
            'rationale': 'Gradient flow 개선 및 특성 재사용으로 수렴 가속화',
            'architecture_sketch': {
                'connections': ' 각 stage에서 이전 feature maps를 concatenate',
                'bottleneck': 'Channel reduction을 통한 효율성 유지',
                'pooling': 'Dense 연결로 spatial dimension 유지',
            },
            'expected_benefit': '더 깊은 학습 경로와 빠른 수렴',
            'implementation_file': 'models/dense_residual_cnn.py',
            'config_changes': {
                'model': {
                    'name': 'dense_residual_cnn',
                    'params': {'use_dense_connections': True, 'bottleneck_ratio': 0.5}
                }
            }
        })

    # 제안 4: Hybrid Approach - CNN + Local Transformer
    proposals.append({
        'type': 'model',
        'name': 'LocalTransformerCNN',
        'description': 'Local window attention (효율적)을 사용한 CNN + Transformer 하이브리드',
        'rationale': 'Full self-attention 비용 제거하며 long-range dependencies 학습',
        'architecture_sketch': {
            'blocks': '8x8 local windows에서만 attention 수행',
            'efficiency': 'Quadratic attention → Linear complexity',
            'fusion': 'CNN features를 local transformer로 정제',
        },
        'expected_benefit': '빠른 학습속도 + attention의 장점 활용',
        'implementation_file': 'models/local_transformer_cnn.py',
        'config_changes': {
            'model': {
                'name': 'local_transformer_cnn',
                'params': {'window_size': 8, 'num_heads': 4}
            }
        }
    })

    # 제안 5: Spectral 구조 활용 - Fourier Features
    proposals.append({
        'type': 'model',
        'name': 'SpectralAwareCNN',
        'description': '입력의 spectral 특성을 활용한 주기성 인식 CNN',
        'rationale': '128x128은 정기적 구조 → Fourier space에서의 전처리로 효율성 증대',
        'architecture_sketch': {
            'frontend': '입력에 FFT 추가 (학습 가능한 주파수 필터)',
            'frequency_encoding': 'Positional encoding with frequency information',
            'backbone': '표준 CNN backbone',
        },
        'expected_benefit': '주기적 구조 명시적 학습으로 성능 향상',
        'implementation_file': 'models/spectral_aware_cnn.py',
        'config_changes': {
            'model': {
                'name': 'spectral_aware_cnn',
                'params': {'use_fft_encoding': True, 'fft_bins': 32}
            }
        }
    })

    return proposals


def propose_creative_losses(log_data: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict]:
    """창의적인 손실 함수 제안"""
    proposals = []
    current_loss = log_data['loss_name']
    current_val_loss = log_data['best_val_loss']

    # 제안 1: Huber Loss + Correlation (robust + structure)
    proposals.append({
        'type': 'loss',
        'name': 'HuberPearsonLoss',
        'description': 'Huber loss (outlier robust)와 Pearson correlation (구조 학습) 결합',
        'rationale': '현재 MSE_Pearson에서 MSE를 Huber로 교체하여 outlier에 덜 민감',
        'formula': 'L = 0.8 * Huber(pred, target) + 0.2 * (1 - Pearson correlation)',
        'expected_benefit': 'Outliers에 robust하면서도 spectral shape 유지',
        'implementation_file': 'losses/huber_pearson.py',
        'config_changes': {
            'loss': {
                'name': 'huber_pearson',
                'params': {'huber_delta': 0.5, 'pearson_weight': 0.2}
            }
        }
    })

    # 제안 2: TV(Total Variation) + MSE - Smooth predictions
    proposals.append({
        'type': 'loss',
        'name': 'SmoothnessMSELoss',
        'description': 'MSE + Total Variation (인접 bin 간의 차이 최소화)',
        'rationale': '스펙트럼의 부드러운 변화 강제로 노이즈 감소 및 일반화 향상',
        'formula': 'L = 0.7 * MSE(pred, target) + 0.3 * TV(pred)',
        'tv_definition': '인접 bins의 절대값 차이의 합',
        'expected_benefit': '덜 noisy한 스펙트럼 예측, 더 부드러운 곡선',
        'implementation_file': 'losses/smoothness_mse.py',
        'config_changes': {
            'loss': {
                'name': 'smoothness_mse',
                'params': {'mse_weight': 0.7, 'tv_weight': 0.3}
            }
        }
    })

    # 제안 3: Quantile Loss - 특정 범위 강조
    proposals.append({
        'type': 'loss',
        'name': 'QuantileRobustLoss',
        'description': '하위 50%와 상위 10%를 다르게 가중치주는 Quantile loss',
        'rationale': '낮은 값의 정확성은 덜 중요하고 높은 값은 더 정확히',
        'formula': 'L = quantile_loss(pred, target, quantile=0.5) + 2*quantile_loss(pred, target, quantile=0.9)',
        'expected_benefit': '실제 중요한 부분(높은 intensity)에 더 집중',
        'implementation_file': 'losses/quantile_robust.py',
        'config_changes': {
            'loss': {
                'name': 'quantile_robust',
                'params': {'lower_quantile': 0.5, 'upper_quantile': 0.9, 'weight_ratio': 2.0}
            }
        }
    })

    # 제안 4: Wasserstein Loss - Distribution matching
    if current_val_loss > 0.02:
        proposals.append({
            'type': 'loss',
            'name': 'WassersteinSpectralLoss',
            'description': 'Wasserstein distance로 spectrum distribution 매칭',
            'rationale': '절대값이 아닌 분포 유사성 학습으로 더 robust한 학습',
            'formula': 'L = W(pred_dist, target_dist) + 0.1 * Pearson correlation',
            'expected_benefit': '분포 수준의 매칭으로 더 안정적 학습',
            'implementation_file': 'losses/wasserstein_spectral.py',
            'config_changes': {
                'loss': {
                    'name': 'wasserstein_spectral',
                    'params': {'num_bins': 30, 'correlation_weight': 0.1}
                }
            }
        })

    # 제안 5: Multi-scale Loss - Frequency domain까지 고려
    proposals.append({
        'type': 'loss',
        'name': 'MultiScaleSpectralLoss',
        'description': '원본 + smoothed versions에 대해 동시 학습',
        'rationale': '다양한 스케일에서의 정확성 동시 달성',
        'formula': 'L = MSE(pred, target) + 0.5*MSE(smooth(pred), smooth(target)) + 0.5*Pearson',
        'smoothing': '가우시안 필터로 bin dimension smoothing',
        'expected_benefit': '다양한 주파수 대역에서 균형잡힌 학습',
        'implementation_file': 'losses/multiscale_spectral.py',
        'config_changes': {
            'loss': {
                'name': 'multiscale_spectral',
                'params': {'smooth_kernel': 3, 'pearson_weight': 0.5}
            }
        }
    })

    return proposals


def generate_proposal_summary(model_proposals: List[Dict], loss_proposals: List[Dict]) -> str:
    """제안 요약 생성"""
    summary = "\n" + "=" * 100 + "\n"
    summary += "🎯 CREATIVE MODEL & LOSS FUNCTION PROPOSALS\n"
    summary += "=" * 100 + "\n"

    summary += "\n📊 NEW MODEL ARCHITECTURES:\n"
    summary += "-" * 100 + "\n"
    for i, prop in enumerate(model_proposals, 1):
        summary += f"\n{i}. {prop['name']}\n"
        summary += f"   📝 {prop['description']}\n"
        summary += f"   💡 Why: {prop['rationale']}\n"
        summary += f"   📈 Expected: {prop['expected_benefit']}\n"
        summary += f"   📄 File: {prop['implementation_file']}\n"

    summary += "\n\n💔 NEW LOSS FUNCTIONS:\n"
    summary += "-" * 100 + "\n"
    for i, prop in enumerate(loss_proposals, 1):
        summary += f"\n{i}. {prop['name']}\n"
        summary += f"   📝 {prop['description']}\n"
        summary += f"   💡 Why: {prop['rationale']}\n"
        summary += f"   📐 Formula: {prop['formula']}\n"
        summary += f"   📈 Expected: {prop['expected_benefit']}\n"
        summary += f"   📄 File: {prop['implementation_file']}\n"

    summary += "\n\n" + "=" * 100 + "\n"
    summary += "🚀 NEXT STEPS:\n"
    summary += "=" * 100 + "\n"
    summary += """
1. 관심있는 모델/손실 선택
2. 해당 구현 파일 작성 (모습: models/*.py 또는 losses/*.py)
3. Config 파일 생성 (test_config_YYYY.yaml)
4. 학습 실행: python CR_recon/train.py --config test_config_YYYY.yaml
5. Loss 값 비교 및 채택

💡 팁: 여러 조합을 동시에 테스트해보세요 (e.g., 새 모델 + 새 손실함수)
"""

    return summary


def create_implementation_stubs(proposals: List[Dict], output_dir: Path):
    """구현 스텁 생성 (실제 구현은 사용자가 하도록)"""
    print("\n📝 Generating implementation stubs...\n")

    for proposal in proposals:
        file_path = output_dir / proposal['implementation_file']
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 이미 있으면 스킵
        if file_path.exists():
            print(f"  ✓ {proposal['implementation_file']} (already exists)")
            continue

        if proposal['type'] == 'model':
            stub = _create_model_stub(proposal)
        else:
            stub = _create_loss_stub(proposal)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(stub)
        print(f"  ✓ {proposal['implementation_file']} created")


def _create_model_stub(proposal: Dict) -> str:
    """모델 구현 스텁 생성"""
    return f'''"""
{proposal['name']}: {proposal['description']}

아키텍처:
{proposal['architecture_sketch']}

기대 효과: {proposal['expected_benefit']}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class {proposal['name']}(nn.Module):
    """
    {proposal['description']}

    입력: (B, 1, 128, 128)
    출력: (B, 2, 2, 30)
    """

    def __init__(self, out_len=30, d_model=192, **kwargs):
        """
        Args:
            out_len: Output spectrum bins
            d_model: Feature dimension
            **kwargs: Additional parameters from config
        """
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model

        # TODO: 아키텍처 구현
        # {proposal['name']}의 설계 철학:
        # {proposal['rationale']}

        # Stem (128 → 64)
        # self.stem_conv = ...

        # Backbone stages
        # self.stage1 = ...
        # self.stage2 = ...
        # self.stage3 = ...
        # self.stage4 = ...

        # Head
        # self.head = ...

        raise NotImplementedError(f"{{self.__class__.__name__}} 구현 필요")

    def forward(self, x):
        """
        x: (B, 1, 128, 128)
        Returns: (B, 2, 2, out_len)
        """
        # TODO: Forward pass 구현
        raise NotImplementedError()
'''


def _create_loss_stub(proposal: Dict) -> str:
    """손실 함수 구현 스텁 생성"""
    return f'''"""
{proposal['name']}: {proposal['description']}

공식: {proposal['formula']}

기대 효과: {proposal['expected_benefit']}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class {proposal['name']}(nn.Module):
    """
    {proposal['description']}
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Loss parameters from config
        """
        super().__init__()
        # TODO: 파라미터 저장
        # self.param1 = kwargs.get('param1', default_value)
        raise NotImplementedError(f"{{self.__class__.__name__}} 구현 필요")

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 4, out_len) 또는 (B, 2, 2, out_len)
            target: pred와 동일한 shape

        Returns:
            loss: scalar
        """
        # TODO: 손실 함수 계산
        # 기본 구조:
        # 1. pred와 target을 원하는 shape로 reshape
        # 2. 손실 계산 (예: {proposal['formula']})
        # 3. 결과 반환

        raise NotImplementedError()
'''


def print_analysis(log_data: Dict[str, Any], analysis: Dict[str, Any]):
    """현재 성능 분석 출력"""
    print("\n" + "=" * 100)
    print("📊 CURRENT PERFORMANCE ANALYSIS")
    print("=" * 100)

    print(f"\n🎯 Current Configuration:")
    print(f"  Model: {log_data['model_name']}")
    print(f"  Loss: {log_data['loss_name']}")
    print(f"  Best Val Loss: {log_data['best_val_loss']:.6e}")
    print(f"  Epochs Trained: {log_data['epochs_trained']}")

    print(f"\n📈 Performance Metrics:")
    print(f"  Improvement Rate: {analysis['metrics'].get('improvement_rate', 0):.2%}")
    print(f"  Convergence Trend: {analysis['metrics'].get('convergence_trend', 'unknown')}")
    print(f"  Train-Val Gap: {analysis['metrics'].get('train_val_gap', 0):.2%}")
    print(f"  Overfitting Status: {analysis['metrics'].get('overfitting', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description='창의적 모델/손실 함수 제안')
    parser.add_argument('--log', required=True, help='훈련 로그 파일 경로')
    parser.add_argument('--base-config', default='CR_recon/configs/default.yaml',
                        help='Base config 파일 경로')
    args = parser.parse_args()

    # 로그 파싱
    print(f"\n📖 Parsing log file: {args.log}")
    log_data = parse_train_log(args.log)

    # 성능 분석
    print(f"📊 Analyzing performance...")
    analysis = analyze_performance(log_data)

    # 분석 결과 출력
    print_analysis(log_data, analysis)

    # 창의적 제안 생성
    print(f"\n💡 Generating creative proposals...")
    model_proposals = propose_creative_models(log_data, analysis)
    loss_proposals = propose_creative_losses(log_data, analysis)

    # 제안 요약 출력
    summary = generate_proposal_summary(model_proposals, loss_proposals)
    print(summary)

    # 구현 스텁 생성
    output_dir = Path('CR_recon')
    create_implementation_stubs(model_proposals + loss_proposals, output_dir)

    # 전체 제안 내용을 JSON으로 저장 (나중 참고용)
    proposals_json = {
        'log_analysis': {
            'model': log_data['model_name'],
            'loss': log_data['loss_name'],
            'best_val_loss': float(log_data['best_val_loss']),
            'epochs_trained': log_data['epochs_trained'],
        },
        'analysis': analysis,
        'model_proposals': model_proposals,
        'loss_proposals': loss_proposals,
    }

    proposals_file = Path('CR_recon/outputs/creative_proposals.json')
    proposals_file.parent.mkdir(parents=True, exist_ok=True)
    with open(proposals_file, 'w', encoding='utf-8') as f:
        json.dump(proposals_json, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Proposals saved to: {proposals_file}")


if __name__ == '__main__':
    main()
