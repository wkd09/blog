---
title: "mHC 논문 정리: Residual Stream을 넓히면서 학습을 안정화하기"
date: 2026-11-23 00:00:00 +0900
summary: "Hyper-Connections의 표현력과 학습 불안정성을 살펴보고, 이중 확률 행렬과 Sinkhorn-Knopp 알고리즘으로 residual 경로를 제약하는 mHC를 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - Transformer
  - DeepSeek
  - Paper
source: "arXiv:2512.24880"
---

Residual connection은 깊은 신경망에서 입력이 여러 층을 지나도 직접 전달될 수 있게 한다. 그렇다면 이 경로를 여러 개로 넓히고 서로 정보를 섞으면 어떨까? **Hyper-Connections(HC)**는 이 아이디어로 성능을 높이지만, 자유롭게 학습되는 연결이 쌓이면서 신호와 기울기가 불안정해질 수 있다.

DeepSeek-AI의 [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)는 residual 경로의 혼합 행렬에 제약을 걸어 이 문제를 다룬다. 이 글은 2026년 1월 5일 공개된 논문 v2를 기준으로 정리했다. 그림과 표는 논문에서 가져왔다.

![일반 residual connection, HC, mHC의 구조 비교]({{ '/assets/images/blog/mhc-paper/01.png' | relative_url }})

## 1. Residual connection에서 HC까지

일반적인 residual connection은 층의 입력을 변환 결과에 그대로 더한다.

$$
x_{l+1}=x_l+\mathcal{F}(x_l, W_l)
$$

여러 층을 펼쳐 보면 처음 입력이 더해지는 경로가 남는다. 이 **항등 경로**는 층을 지나는 동안 입력과 기울기가 직접 전달되는 통로다.

HC는 하나의 residual stream을 $n$개로 확장한다. 각 stream의 차원이 $C$라면 전체 상태는 $x_l\in\mathbb{R}^{n\times C}$다. 층 자체는 여전히 $C$차원 입력을 받으며, 세 가지 학습 가능한 매핑이 stream을 관리한다.

$$
x_{l+1}=H_l^{\mathrm{res}}x_l + (H_l^{\mathrm{post}})^\top\mathcal{F}(H_l^{\mathrm{pre}}x_l,W_l)
$$

| 매핑 | 역할 |
|---|---|
| $H^{\mathrm{pre}}$ | 여러 stream을 모아 층의 입력을 만든다 |
| $H^{\mathrm{post}}$ | 층의 출력을 각 stream에 전달한다 |
| $H^{\mathrm{res}}$ | residual 경로 안에서 stream끼리 섞는다 |

층의 폭을 그대로 두면서 residual stream의 정보 용량과 연결 방식을 늘리는 것이 핵심이다. 아래 제거 실험에서는 고정 매핑을 사용한 기준 대비 residual 매핑만 학습해도 손실이 0.022 낮아졌다. 세 매핑을 모두 학습했을 때 차이는 0.027이었다. 이 실험에서는 stream 사이의 정보 교환이 성능 개선에 크게 기여했다.

![HC의 세 매핑에 대한 제거 실험 결과]({{ '/assets/images/blog/mhc-paper/02.png' | relative_url }})

## 2. 넓어진 경로가 불안정해지는 이유

일반 residual connection의 직접 경로에는 항등 행렬이 있다. 반면 HC에서 여러 층을 통과하는 residual 경로에는 $H_l^{\mathrm{res}}$ 행렬들이 연속해서 곱해진다. 각 행렬이 제약 없이 학습되므로 곱이 신호를 크게 증폭하거나 약화할 수 있다.

논문의 27B 실험에서 HC는 약 12,000 step 이후 mHC 대비 손실이 상승했고 gradient norm도 불안정하게 움직였다. 이 학습 곡선은 뒤의 실험 결과에서 기준 모델과 함께 비교한다. 연구진은 원인을 살펴보기 위해 행렬의 절댓값 행 합과 열 합의 최대값을 이용해 각각 순전파와 역전파의 최대 증폭 정도를 측정했다.

![HC의 단일 층과 복합 residual 매핑의 신호 및 기울기 증폭]({{ '/assets/images/blog/mhc-paper/04.png' | relative_url }})

개별 층의 값은 대체로 1 근처여도 여러 층의 행렬을 곱하면 역전파 방향의 최대 gain이 약 3,000까지 커졌다. 작은 변화가 깊이에 따라 누적될 수 있다는 뜻이다.

## 3. mHC: 섞되 평균을 보존한다

mHC는 $H^{\mathrm{res}}$를 **이중 확률 행렬(doubly stochastic matrix)**로 제한한다. 원소가 음수가 아니며, 각 행과 각 열의 합이 1인 행렬이다.

$$
H^{\mathrm{res}}\mathbf{1}=\mathbf{1},\qquad
\mathbf{1}^{\top}H^{\mathrm{res}}=\mathbf{1}^{\top},\qquad
H^{\mathrm{res}}\geq 0
$$

각 출력 stream은 입력 stream의 볼록 결합이 된다. 열의 합도 1이므로 stream 전체의 평균이 보존된다. 이중 확률 행렬끼리 곱해도 다시 이중 확률 행렬이고, 스펙트럼 노름은 1을 넘지 않는다. 따라서 이상적인 residual 혼합 경로는 깊어져도 그 자체로 신호를 증폭하지 않는다.

이 제약이 모든 stream을 그대로 전달한다는 뜻은 아니다. 항등 행렬도 허용되지만, 다른 이중 확률 행렬은 stream을 섞는다. 또 이 성질은 **residual 혼합 경로**에 대한 것으로, 모델 전체의 기울기가 항상 안정적이거나 정보 손실이 없다는 보장은 아니다. 예를 들어 모든 원소가 $1/n$인 행렬은 stream 사이의 차이를 평균으로 지운다.

이 행렬들의 집합은 순열 행렬들의 볼록 껍질인 **Birkhoff polytope**로 볼 수 있다. 논문은 양수 행렬의 행과 열을 번갈아 정규화하는 Sinkhorn-Knopp 알고리즘을 사용해 이 조건에 근사한다. 실제 실험에서는 20회 반복했다. 입력 매핑에는 sigmoid, 출력 매핑에는 $2\,\mathrm{sigmoid}$를 적용해 음수가 되지 않도록 한다.

![mHC의 단일 층과 복합 매핑에서 관측한 신호 및 기울기 gain]({{ '/assets/images/blog/mhc-paper/09.png' | relative_url }})

유한 횟수로 정규화하기 때문에 실제 행렬은 조건을 정확히 만족하지 않는다. 위 27B 실험에서 복합 매핑의 역전파 gain은 최대 약 1.6이었다. HC의 약 3,000과 비교하면 크게 줄어든 수치다.

## 4. 연산량만으로는 알 수 없는 비용

Residual stream을 $n$개로 넓히면 추가 매핑의 FLOPs가 상대적으로 작더라도 메모리 접근과 통신량이 늘어난다. 아래 표는 층 내부 연산을 제외한, 토큰당 순전파 residual 유지 비용이다.

![일반 residual connection과 HC의 토큰당 메모리 읽기 및 쓰기 비용]({{ '/assets/images/blog/mhc-paper/05.png' | relative_url }})

논문은 이 비용을 줄이기 위해 여러 연산을 하나의 커널로 합치고, 역전파에 필요한 일부 중간 결과를 저장하는 대신 다시 계산한다. 또한 파이프라인 병렬 처리에서 늘어난 통신을 계산과 겹치도록 DualPipe 스케줄을 조정한다. 연구진의 내부 대규모 학습에서는 $n=4$일 때 추가 학습 시간이 6.7%였다고 보고했다. 이는 해당 시스템과 최적화에서의 측정치다.

## 5. 실험 결과

연구진은 DeepSeek-V3에서 영감을 받은 MoE 모델을 사용해 기본 residual connection, HC, mHC를 비교했다. HC와 mHC의 stream 확장률은 모두 4다. 주요 결과는 27B 모델이며, 추가로 3B와 9B에서 규모에 따른 변화를 살펴봤다.

![27B 모델의 기준 대비 학습 손실 차이와 gradient norm]({{ '/assets/images/blog/mhc-paper/06.png' | relative_url }})

27B 실험의 마지막 학습 손실은 mHC가 기준 모델보다 0.021 낮았다. HC에 비해 손실과 gradient norm의 변동도 작았다. 다만 기준 모델의 gradient norm과 완전히 같다는 의미는 아니다.

![27B 모델의 8개 벤치마크 결과 비교]({{ '/assets/images/blog/mhc-paper/07.png' | relative_url }})

mHC는 8개 벤치마크 모두에서 기준 모델보다 높았고, 그중 7개에서 HC보다 높았다. 예를 들어 BBH는 기준 43.8, HC 48.9, mHC 51.0이고 DROP은 각각 47.0, 51.6, 53.9다. MATH에서는 HC가 26.4로 mHC의 26.0보다 높았다. 각 벤치마크의 지표와 shot 수는 표에서 다르므로 숫자를 서로 직접 비교하기보다 같은 열 안에서 비교해야 한다.

![연산량 규모와 학습 토큰 수에 따른 기준 대비 mHC 손실 차이]({{ '/assets/images/blog/mhc-paper/08.png' | relative_url }})

3B, 9B, 27B에 걸친 연산량 비교와 3B 모델의 토큰 규모 실험에서도 mHC의 손실은 기준보다 낮았다. 이는 논문에서 측정한 범위에서의 결과이며, 모든 모델과 학습 조건으로 일반화할 수 있는 증거는 아니다.

## 정리

HC는 residual stream을 넓히고 학습 가능한 연결로 표현력을 늘린다. mHC는 그중 residual 혼합 행렬을 이중 확률 행렬에 가깝게 제한해 stream 간 정보 교환과 안정적인 신호 전달을 함께 추구한다. 논문의 실험에서는 이 제약이 HC의 불안정성을 완화하면서 기준 모델 대비 성능 이점을 유지했다.

논문: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) · [PDF](https://arxiv.org/pdf/2512.24880)
