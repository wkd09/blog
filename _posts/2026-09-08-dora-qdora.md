---
title: "DoRA와 QDoRA 정리: Weight Decomposition으로 LoRA의 한계와 양자화 비용 다루기"
date: 2026-09-08 00:00:00 +0900
last_modified_at: 2026-09-08 00:00:00 +0900
summary: "DoRA가 weight의 magnitude와 direction을 분리해 LoRA의 표현력을 높이는 방법과, 이를 양자화 환경으로 확장한 QDoRA·DL-QAT의 차이를 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - LoRA
  - DoRA
  - QDoRA
  - Quantization
  - QAT
  - PEFT
  - Paper
source: "arXiv:2402.09353, arXiv:2504.09223"
---

# DoRA와 QDoRA: LoRA를 weight decomposition과 양자화로 확장하기

이 글은 두 논문을 바탕으로 정리한 글이다.

- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2504.09223)

먼저 용어를 하나 짚고 넘어가자. 두 번째 링크의 논문이 제안하는 공식 방법명은 **QDoRA가 아니라 DL-QAT**다. 실무나 커뮤니티에서 `QDoRA(Quantized DoRA)`는 보통 QLoRA처럼 양자화한 base model 위에 DoRA adapter를 학습하는 조합을 뜻한다. 반면 DL-QAT는 DoRA의 핵심 아이디어인 magnitude 분리를 양자화-aware training에 맞게 다시 설계한 방법이다.

둘은 완전히 같은 방법은 아니지만, 문제의식은 이어져 있다.

> LoRA가 학습해야 하는 weight 변화를 더 구조적으로 나누고, 그중 중요한 크기 조절을 별도로 학습하자.

DoRA는 이 아이디어로 LoRA와 full fine-tuning 사이의 표현력 차이를 줄인다. QDoRA와 DL-QAT는 여기에 양자화를 결합해, 적은 메모리로 학습하면서도 최종 모델을 낮은 bit로 유지하려고 한다.

![DoRA의 weight decomposition 개요](/assets/images/blog/dora-overview.png)

*DoRA는 사전학습 weight를 magnitude와 direction으로 나누고, direction 쪽은 low-rank update로 적응시킨 뒤 학습이 끝나면 다시 하나의 weight로 merge한다. 출처: DoRA 논문 Figure 1.*

## 1. LoRA는 왜 충분하지 않을 수 있는가?

Full fine-tuning에서는 사전학습 weight $W_0$ 전체를 업데이트한다.

$$
W = W_0 + \Delta W
$$

LoRA는 변화량 $\Delta W$가 낮은 rank로 표현될 수 있다고 가정한다.

$$
\Delta W = BA
$$

따라서 실제 weight는 다음처럼 계산된다.

$$
W = W_0 + BA
$$

여기서 $A$와 $B$만 학습하고 $W_0$는 고정한다. 이 방식은 학습 파라미터, optimizer state, checkpoint 크기를 크게 줄인다. 학습이 끝난 뒤에는 $W_0+BA$를 하나의 weight로 병합할 수 있어 inference에 별도 adapter branch도 필요하지 않다.

하지만 LoRA에는 한 가지 중요한 제약이 있다.

> 하나의 low-rank update $BA$가 weight의 방향 변화와 크기 변화를 동시에 표현해야 한다.

논문은 이 제약이 단순히 rank가 작아서 생기는 문제만은 아니라고 분석한다. Weight를 각 column vector의 크기와 방향으로 나누어 보면, full fine-tuning과 LoRA의 업데이트 패턴이 다르다.

- Full fine-tuning은 크기는 크게 바꾸면서 방향은 조금 바꾸거나, 반대로 방향은 크게 바꾸면서 크기는 거의 유지하는 식의 유연한 업데이트를 보인다.
- LoRA는 magnitude 변화와 direction 변화가 함께 움직이는 경향이 강하다.

즉 LoRA는 적은 파라미터로 효율적으로 움직이지만, weight를 세밀하게 조절하는 방식에서는 full fine-tuning보다 덜 자유로울 수 있다.

![Full fine-tuning, LoRA, DoRA의 magnitude·direction 변화 패턴](/assets/images/blog/dora-update-pattern.png)

*Full fine-tuning과 DoRA는 magnitude 변화와 direction 변화가 반대 방향으로 움직이는 경향을 보이는 반면, LoRA는 두 변화가 함께 커지는 패턴을 보인다. 출처: DoRA 논문 Figure 2.*

## 2. DoRA의 핵심: weight를 크기와 방향으로 분리하기

DoRA는 weight matrix를 magnitude와 direction으로 분해한다. 직관적으로 각 column vector를 다음처럼 생각할 수 있다.

$$
w = m \frac{v}{\|v\|}
$$

- $m$: weight vector의 크기, 즉 magnitude
- $v/\|v\|$: weight vector의 방향, 즉 direction

행렬 단위로 쓰면 DoRA의 weight는 다음과 같이 표현된다.

$$
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

여기서 $\|\cdot\|_c$는 각 column vector에 대한 norm이다. DoRA는 다음 두 종류의 파라미터를 학습한다.

1. column별 magnitude vector $m$
2. direction을 바꾸는 low-rank matrix $A$, $B$

학습 시작 시점에는 $m=\|W_0\|_c$로 초기화하고, $BA=0$이 되도록 초기화한다. 따라서 처음에는 DoRA의 weight가 원래 weight $W_0$와 동일하다.

```text
LoRA:
    W = W0 + BA

DoRA:
    W = magnitude × normalize(W0 + BA)
      = m × (W0 + BA) / column_norm(W0 + BA)
```

핵심은 direction 업데이트를 그대로 full-rank로 학습하지 않는다는 점이다. 파라미터 수가 큰 방향 성분은 LoRA의 low-rank update로 처리하고, 상대적으로 작은 magnitude vector는 직접 학습한다.

## 3. DoRA가 LoRA보다 안정적인 이유

DoRA는 단순히 파라미터를 더 추가한 LoRA가 아니다. weight normalization과 비슷한 reparameterization을 통해 optimization의 조건을 개선한다.

DoRA의 방향 성분에 대한 gradient는 대략 다음 두 효과를 가진다.

- 현재 weight 방향과 겹치는 성분을 projection으로 제거한다.
- magnitude와 direction의 비율에 따라 gradient 크기를 조절한다.

이렇게 하면 direction update가 weight의 크기를 불필요하게 키우는 방향으로 끌려가는 현상을 줄일 수 있다. 논문은 이 효과가 LoRA direction update의 학습 안정성을 높인다고 설명한다.

또한 DoRA의 학습 결과는 LoRA보다 full fine-tuning의 패턴에 가까웠다. 논문 분석에서 direction 변화와 magnitude 변화의 상관관계는 다음과 같이 보고된다.

| 방법 | magnitude와 direction 변화의 상관관계 |
| --- | ---: |
| Full fine-tuning | -0.62 |
| LoRA | 0.83 |
| DoRA | -0.31 |

이 수치는 DoRA가 full fine-tuning처럼 크기와 방향을 독립적으로 조절하는 경향을 어느 정도 회복했다는 근거로 제시된다.

## 4. DoRA 실험 결과

DoRA 논문은 LLaMA, LLaMA2, LLaMA3뿐 아니라 LLaVA와 VL-BART 같은 vision-language 모델에서도 LoRA와 비교한다.

LLaMA-7B의 commonsense reasoning 평균 점수는 다음과 같다.

| 방법 | 학습 파라미터 비율 | 평균 점수 |
| --- | ---: | ---: |
| LoRA | 0.83% | 74.7 |
| DoRA, rank 동일 | 0.84% | 78.4 |
| DoRA, rank 절반 | 0.43% | 77.5 |

같은 rank에서는 DoRA가 LoRA보다 높은 점수를 기록했고, rank를 절반으로 줄여도 LoRA보다 높았다. 즉 DoRA는 단순히 더 많은 학습 파라미터를 사용해서 좋아진 것이 아니라, 같은 parameter budget에서 weight를 나누어 학습하는 방식 자체의 이점을 보였다.

멀티모달 실험에서도 비슷한 경향이 나타난다. VL-BART의 image-text task에서는 DoRA가 LoRA보다 약 1%, video-text task에서는 약 2% 높은 평균 성능을 보였다. LLaVA-1.5-7B의 visual instruction tuning에서도 DoRA는 LoRA보다 0.7% 높은 평균 점수를 기록했다.

그리고 중요한 배포상의 장점이 있다.

> 학습이 끝난 DoRA weight는 다시 하나의 weight matrix로 merge할 수 있으므로, LoRA와 마찬가지로 추가적인 inference latency를 만들지 않는다.

## 5. QDoRA는 무엇인가?

DoRA가 정확도와 표현력을 높여주지만, 모델을 학습할 때 base model의 weight와 optimizer state를 저장해야 한다는 비용은 여전히 남아 있다. 이 문제를 줄이기 위해 base model을 낮은 bit로 양자화하고 DoRA adapter를 학습하는 방식이 QDoRA다.

```text
Full fine-tuning:
    FP16/BF16 base model 전체를 학습

LoRA:
    FP16/BF16 base model 고정 + LoRA A/B 학습

QLoRA:
    4-bit base model 고정 + 고정밀 LoRA A/B 학습

QDoRA:
    4-bit base model 고정 + DoRA magnitude와 low-rank direction 학습
```

QLoRA와 QDoRA의 차이는 adapter의 parameterization에 있다.

- QLoRA는 양자화된 base weight 위에 $BA$를 더한다.
- QDoRA는 양자화된 base weight에 대해 magnitude와 direction을 분리해 학습한다.

따라서 QDoRA는 QLoRA의 메모리 효율을 유지하면서, DoRA가 제공하는 추가적인 크기 조절 능력을 기대하는 방법이다.

다만 여기서 주의할 점이 있다. 양자화된 weight 자체는 연속적인 FP16 weight가 아니다. weight를 4-bit나 3-bit 값으로 표현하면 가능한 값이 제한되고, rounding과 clipping 오차가 생긴다. 따라서 DoRA의 magnitude를 그대로 적용한다고 해서 항상 원래 DoRA와 같은 최적화가 이루어지는 것은 아니다.

이 지점에서 두 번째 논문인 DL-QAT가 등장한다.

## 6. DL-QAT: DoRA 아이디어를 QAT에 맞게 다시 설계하기

두 번째 논문은 양자화-aware training을 parameter-efficient하게 수행하는 방법을 제안한다. 목표는 다음과 같다.

- 낮은 bit weight로 실제 inference 조건을 반영한다.
- full-model QAT처럼 모든 weight를 학습하지 않는다.
- DoRA처럼 weight의 크기 조절과 low-rank update를 분리한다.

일반적인 양자화에서는 floating-point weight $W$를 scale $s$와 bias $b$를 사용해 quantized weight로 바꾼다.

$$
\tilde{W} = clip\left(\left\lfloor \frac{W-b}{s} \right\rceil\right)
$$

$$
W_q = s\tilde{W}+b
$$

QAT는 forward pass에서 $W_q$를 사용하고, rounding처럼 미분하기 어려운 연산은 Straight-Through Estimator, 즉 STE로 gradient를 흘려보낸다.

![DL-QAT의 weight-decomposed quantization 수식](/assets/images/blog/dlqat-weight-decomposed-quantization.png)

*DL-QAT는 양자화된 LoRA weight에 quantization group별 magnitude $m$을 곱해, group-specific magnitude 학습과 정해진 quantization space 안의 weight fine-tuning을 분리한다. 출처: DL-QAT 논문 Section 3.2, Equation 6.*

DL-QAT는 여기에 quantization group별 magnitude $m$을 추가한다.

$$
W_q = m \cdot (W_0 + \alpha BA)_q
$$

이 식을 두 부분으로 나누어 생각할 수 있다.

1. $m$: 각 quantization group의 전체 크기를 조절
2. $A$, $B$: 정해진 quantization space 안에서 weight를 미세 조정

논문이 제안하는 학습 흐름은 다음과 같다.

```text
1. s와 b로 초기 quantization space를 정한다.
2. 초기 단계에서 s와 b를 학습해 양자화 범위를 안정화한다.
3. s와 b를 고정한다.
4. group별 magnitude m과 LoRA A/B를 학습한다.
5. forward에서는 계속 quantized weight를 사용한다.
```

여기서 `group-specific magnitude`가 DoRA의 magnitude vector와 대응한다. 다만 일반 DoRA처럼 각 column weight를 직접 스케일하는 것이 아니라, 양자화 group 단위로 스케일을 학습한다. 양자화된 공간에 맞게 magnitude의 자유도를 줄인 셈이다.

## 7. QDoRA와 DL-QAT는 어떻게 다른가?

둘은 관련 있지만 같은 이름의 동일한 알고리즘으로 보면 안 된다.

| 구분 | QDoRA | DL-QAT |
| --- | --- | --- |
| 기본 아이디어 | 양자화된 base model + DoRA adapter | weight-decomposed low-rank QAT |
| 양자화 방식 | 보통 QLoRA 계열의 4-bit base model | forward에서 quantization을 시뮬레이션하고 QAT 수행 |
| 학습 대상 | DoRA magnitude와 low-rank direction | group magnitude $m$과 LoRA $A,B$ |
| 목적 | 낮은 메모리로 DoRA fine-tuning | 낮은 bit inference weight를 직접 고려하며 QAT |
| 양자화 범위 | 구현과 설정에 따라 다름 | 초기 scale/bias로 정한 quantization space를 사용 |
| 논문 공식 명칭 | 커뮤니티·구현에서 쓰이는 조합명 | DL-QAT |

실무적으로는 다음처럼 이해하면 편하다.

> QDoRA는 “양자화된 모델 위에서 DoRA를 학습하는 큰 범주의 조합”이고, DL-QAT는 “DoRA의 weight decomposition 관점을 양자화-aware training의 최적화 문제에 맞게 구체화한 논문 방법”이다.

DoRA 논문 자체도 결론 부분에서 QDoRA/FSDP 구현과 실험을 언급한다. 따라서 QDoRA는 DoRA와 양자화를 연결하는 실용적인 방향이고, DL-QAT는 그 연결을 quantization space와 group magnitude 관점에서 더 명시적으로 다룬 후속 접근으로 볼 수 있다.

## 8. DL-QAT 실험에서 보이는 효과

DL-QAT 논문은 LLaMA와 LLaMA2 모델에 대해 group-wise와 channel-wise 양자화를 평가한다.

LLaMA-7B, group size 128의 weight-only quantization 결과에서 다음과 같은 평균 점수를 보고한다.

| 방법 | bit | MMLU 0-shot | Common Sense 평균 |
| --- | ---: | ---: | ---: |
| QA-LoRA | 4 | 37.9 | 60.8 |
| DL-QAT | 4 | 40.5 | 63.4 |
| QA-LoRA | 3 | 32.2 | 59.3 |
| DL-QAT | 3 | 36.4 | 61.1 |

논문은 LLaMA-7B 3-bit 설정에서 QA-LoRA보다 MMLU zero-shot 정확도가 4.2% 높았다고 설명한다. LLaMA2-7B 4-bit per-channel 설정에서는 common-sense QA 평균이 QA-LoRA보다 5.5% 높았다.

메모리 측면에서 DL-QAT는 quantization parameter를 초기화한 뒤 $m,A,B$만 학습한다. 논문 표에 따르면 LLaMA-7B의 channel-wise 설정에서 학습 파라미터는 약 1M, GPU memory는 약 41GB였다. LLaMA-13B에서는 약 2M 학습 파라미터와 약 65GB GPU memory를 사용했다.

다만 이 수치는 AMD MI250, batch size 16, quantization simulation이라는 특정 실험 환경의 결과다. 다른 GPU, kernel, optimizer, sequence length, batch size에서는 그대로 재현된다고 보장할 수 없다.

## 9. 세 방법의 차이를 한 번에 보기

```text
                 base weight        학습하는 것              배포 weight

Full FT          FP16/BF16 전체      W 전체                   FP16/BF16 W
LoRA             FP16/BF16 고정      A, B                     W0 + BA
DoRA             FP16/BF16 고정      m, A, B                  m × normalize(W0 + BA)
QLoRA            4-bit 고정          고정밀 A, B               양자화 base + adapter
QDoRA            4-bit 고정          m, A, B                  양자화 base + DoRA 방식
DL-QAT           quantized forward  m, A, B (+ 초기 s, b)    낮은 bit weight를 고려한 모델
```

가장 큰 차이는 **무엇을 고정하고, 어떤 공간에서 update를 학습하는가**다.

- LoRA는 원래 weight 공간에서 low-rank 변화량을 학습한다.
- DoRA는 weight 공간을 magnitude와 direction으로 나눈다.
- QLoRA와 QDoRA는 base weight를 낮은 bit로 저장해 메모리를 줄인다.
- DL-QAT는 forward부터 양자화된 weight를 사용하면서, group magnitude와 low-rank update를 함께 최적화한다.

## 10. 언제 어떤 방법을 선택할까?

### base model을 충분한 GPU 메모리로 학습할 수 있다면

LoRA와 DoRA를 비교해볼 수 있다. 같은 rank와 학습 budget에서 DoRA의 성능이 더 좋거나, 더 작은 rank로 비슷한 성능을 낼 가능성이 있다.

### 학습 메모리가 부족하다면

QLoRA 또는 QDoRA가 현실적인 선택이다. base model을 4-bit로 두고 adapter만 학습하므로 single GPU fine-tuning의 진입 장벽이 낮아진다.

### 최종 배포도 낮은 bit가 중요하다면

단순히 양자화된 base model 위에서 adapter를 학습하는 것만으로는 충분하지 않을 수 있다. adapter를 merge했을 때 다시 고정밀 weight가 되거나, 양자화 오차가 downstream task 성능을 크게 떨어뜨릴 수 있기 때문이다. 이런 경우 QAT 계열, 특히 DL-QAT와 같은 quantization-aware 접근을 검토할 수 있다.

### 정확한 재현성이 중요하다면

방법 이름보다 실제 설정을 확인해야 한다.

- base model의 bit 수
- NF4인지 uniform quantization인지
- group size 또는 channel-wise 설정
- adapter rank와 target module
- magnitude를 column, channel, group 중 어디에 두는지
- quantization scale과 clipping 값을 학습하는지 고정하는지

같은 `QDoRA`라는 이름을 사용해도 이 설정들이 다르면 메모리 사용량과 성능이 크게 달라질 수 있다.

## 11. 한계와 주의점

### DoRA가 항상 LoRA보다 좋은 것은 아니다

DoRA는 magnitude vector를 추가로 저장하고 매 forward에서 normalization 관련 계산을 고려해야 한다. 논문은 merge를 통해 inference overhead를 없앨 수 있다고 설명하지만, 학습 중 메모리와 구현 복잡도는 LoRA보다 커질 수 있다.

### quantization error는 사라지지 않는다

QDoRA는 adapter의 표현력을 높여도 base model의 양자화 오차 자체를 제거하지 않는다. 특히 3-bit처럼 양자화 간격이 크면 clipping과 rounding이 성능을 제한할 수 있다.

### QDoRA라는 이름은 하나의 표준 알고리즘을 뜻하지 않는다

어떤 구현은 QLoRA 위에 DoRA만 추가하고, 어떤 구현은 FSDP와 함께 사용하며, 어떤 구현은 QAT까지 포함한다. 따라서 논문이나 코드에서 `QDoRA`를 봤다면 quantization 방식과 학습 대상 파라미터를 함께 확인해야 한다.

### DL-QAT의 실험은 simulation 환경이다

DL-QAT 논문은 학습 중 learnable parameter를 bf16으로 표현하고 inference에서 quantized weight를 dequantize해 계산하는 실험을 수행한다. 실제 하드웨어에서의 latency와 memory bandwidth 이점은 사용하는 kernel과 runtime에 따라 달라질 수 있다.

## 내가 이해한 핵심

LoRA는 weight 변화량 전체를 low-rank matrix로 표현한다.

```text
LoRA = 변화량을 작게 만들기
```

DoRA는 여기서 한 단계 더 나아가 weight 변화 자체를 두 가지 역할로 나눈다.

```text
DoRA = weight의 크기는 따로 조절하고,
       방향 변화만 low-rank로 학습하기
```

QDoRA는 이 구조를 양자화된 base model에 적용해 메모리를 줄이는 실용적인 조합이다. DL-QAT는 더 구체적으로 양자화 공간을 먼저 정하고, 각 group의 magnitude와 low-rank update를 분리해 학습한다.

결국 세 방법의 흐름은 다음과 같다.

```text
LoRA  → low-rank update
DoRA  → magnitude + low-rank direction
QDoRA/DL-QAT → 위 구조 + low-bit quantization
```

한 줄로 요약하면 다음과 같다.

> DoRA는 LoRA가 weight의 크기와 방향을 한 번에 학습하며 생기는 제약을 분리해 완화하고, QDoRA와 DL-QAT는 그 아이디어를 낮은 bit의 메모리·배포 제약 안으로 가져온 방법이다.

## 참고 자료

- Shih-Yang Liu et al., [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353), ICML 2024.
- Wenjin Ke et al., [DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2504.09223).
- [NVlabs/DoRA 공식 구현](https://github.com/NVlabs/DoRA)
- [Answer.AI의 FSDP QDoRA 글](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html)
