---
title: "Normalization과 PCA 정리"
date: 2026-05-29 13:50:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - study
tags:
  - DeepLearning
  - Normalization
  - PCA
source: "Notion PDF Export - Normalization, PCA"
---

![Normalization and PCA]({{ '/assets/images/blog/normalization-pca.svg' | relative_url }})

<small>Image: [Wikimedia Commons - Gaussian scatter PCA](https://commons.wikimedia.org/wiki/Special:FilePath/GaussianScatterPCA.svg)</small>

Normalization과 PCA는 모두 값의 분포를 다루지만 목적은 다르다.

```text
Normalization
activation이나 feature의 scale을 맞춘다
-> 학습 안정화

PCA
분산이 큰 방향으로 feature space를 다시 만든다
-> 차원 축소
```

이 글에서는 Batch Normalization과 Layer Normalization의 차이, PCA가 principal component를 찾는 이유를 정리한다.

## Batch Normalization

Batch Normalization은 mini-batch 단위로 activation의 평균과 분산을 맞춘다.

![batch normalization]({{ '/assets/images/blog/batch-normalization.png' | relative_url }})
![internal covariate shift]({{ '/assets/images/blog/internal-covariate-shift.png' | relative_url }})

$$
\hat{x}_i
= \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

이후 학습 가능한 $\gamma$, $\beta$를 적용한다.

$$
y_i = \gamma \hat{x}_i + \beta
$$

![normalization formulas]({{ '/assets/images/blog/normalization-formulas.png' | relative_url }})
![batch norm algorithm]({{ '/assets/images/blog/batch-norm-algorithm.png' | relative_url }})

정규화한 값에 학습 가능한 $\gamma$, $\beta$를 다시 적용하므로 model이 필요한 scale과 shift를 복원할 수 있다.

Batch Normalization은 원 논문에서 layer를 통과할 때 activation 분포가 계속 바뀌는 internal covariate shift를 줄이는 방법으로 설명됐다. 실제로는 optimization landscape를 더 안정적으로 만드는 효과 등 여러 관점으로 설명된다. 중요한 것은 batch 통계를 사용한다는 점이다.

Training에서는 현재 mini-batch의 평균과 분산을 사용한다. Inference에서는 학습 중 누적한 running mean과 running variance를 사용한다.

```text
train mode -> 현재 batch statistics
eval mode  -> running statistics
```

이 차이 때문에 inference 전에 `model.eval()`을 호출하지 않으면 결과가 달라질 수 있다.

Batch size가 너무 작으면 평균과 분산 추정이 noisy해질 수 있다. 이 경우 Group Normalization이나 Layer Normalization을 고려할 수 있다.

또한 fine-tuning에서는 BatchNorm layer를 어떻게 다룰지도 중요하다. pretrained CNN을 작은 데이터셋에 맞출 때 running statistics를 계속 업데이트하면 성능이 나빠질 수 있어, BatchNorm을 freeze하는 전략을 쓰기도 한다.

## Layer Normalization

Layer Normalization은 batch 방향이 아니라 sample 하나의 hidden dimension을 기준으로 평균과 분산을 구한다.

각 token 또는 sample 내부에서 계산하므로 다른 sample과 batch size에 의존하지 않는다.

Transformer에서 Layer Normalization을 많이 사용하는 이유도 여기에 있다. Sequence 길이와 batch 구성이 달라져도 같은 방식으로 적용할 수 있다.

Transformer에서는 LayerNorm의 위치도 중요하다.

```text
Post-LN : sub-layer -> residual -> LayerNorm
Pre-LN  : LayerNorm -> sub-layer -> residual
```

깊은 Transformer에서는 Pre-LN이 residual path의 gradient 흐름을 안정적으로 만드는 경우가 많다.

## Batch Norm과 Layer Norm 비교

| 항목 | Batch Normalization | Layer Normalization |
| --- | --- | --- |
| 통계 범위 | 같은 feature의 batch sample | 한 sample의 hidden dimension |
| Batch size 영향 | 큼 | 작음 |
| Train/Eval 통계 | 서로 다름 | 동일한 방식 |
| 대표 사용 | CNN | RNN, Transformer |

CNN에서는 BatchNorm, Transformer에서는 LayerNorm을 먼저 떠올릴 수 있지만 절대적인 규칙은 아니다. Model 구조와 batch size를 함께 봐야 한다.

## PCA

PCA는 Principal Component Analysis의 약자다. 고차원 데이터를 낮은 차원으로 투영하면서 중요한 분산 구조를 최대한 보존한다.

핵심은 데이터의 분산이 가장 큰 방향을 찾는 것이다.

```text
PC1 -> 데이터가 가장 많이 퍼진 방향
PC2 -> PC1과 직교하면서 다음으로 분산이 큰 방향
PC3 -> 앞 component와 직교하는 다음 방향
```

상위 component만 남기면 원래 feature 수보다 적은 축으로 데이터를 표현할 수 있다.

PCA는 다음과 같은 상황에서 유용하다.

- feature 수가 너무 많아 시각화가 어려울 때
- 노이즈가 많은 차원을 줄이고 싶을 때
- 모델 학습 전에 차원을 줄여 계산 비용을 낮추고 싶을 때

PCA는 선형 투영이므로 복잡한 비선형 구조를 보존하지 못할 수 있다. Principal component도 원래 feature의 조합이기 때문에 의미를 바로 해석하기 어렵다.

PCA를 적용하기 전에는 보통 feature scaling을 먼저 한다. PCA는 분산이 큰 방향을 찾기 때문에 단위가 큰 feature가 component를 지배할 수 있다. 예를 들어 키와 연봉처럼 scale이 다른 feature를 그대로 넣으면 연봉 축이 과하게 반영될 수 있다.

PCA 결과에서는 explained variance ratio를 확인한다. 상위 component가 전체 분산의 몇 퍼센트를 설명하는지 보면 몇 차원까지 남길지 정할 수 있다.

## 언제 어떤 방법을 쓸까

Normalization은 model training 중 activation scale을 안정화하는 데 사용한다.

PCA는 training 전 data analysis와 feature preprocessing에 가깝다. Feature 수가 많아 구조를 보기 어렵거나 차원을 줄여 계산량을 낮추고 싶을 때 사용한다.

## 내가 이해한 핵심

Normalization과 PCA는 모두 값의 분포를 바꾸지만 같은 기술이 아니다.

- BatchNorm은 batch statistics로 activation을 정규화한다.
- LayerNorm은 sample 내부 hidden dimension을 정규화한다.
- PCA는 분산이 큰 새로운 축을 찾아 feature dimension을 줄인다.

즉 Normalization은 **network를 안정적으로 학습하는 방법**이고, PCA는 **입력 feature space를 다시 표현하는 방법**이다. 이름이나 수식보다 어느 단계의 어떤 문제를 해결하는지 구분하는 것이 중요하다.
