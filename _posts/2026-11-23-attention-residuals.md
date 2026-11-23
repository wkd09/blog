---
title: "Attention Residuals 논문 정리: 이전 층의 출력을 골라 쓰는 Transformer"
date: 2026-11-23 00:00:00 +0900
summary: "이전 층의 출력을 입력에 따라 가중합하는 Attention Residuals와, 그 비용을 줄이는 Block AttnRes를 살펴본다."
categories:
  - research
tags:
  - AI
  - LLM
  - Transformer
  - Kimi
  - Paper
source: "arXiv:2603.15031"
---

Transformer의 residual connection은 층의 입력에 변환 결과를 더한다. 이 연산을 여러 층에 걸쳐 펼쳐 보면, 토큰 임베딩과 이전 층의 출력이 모두 가중치 1로 더해진다. PreNorm Transformer에서는 깊어질수록 이 합의 크기가 커져 개별 층 출력의 상대적인 기여가 작아질 수 있다. Kimi 팀은 각 층이 이전 출력을 얼마나 반영할지 입력에 따라 정하는 **Attention Residuals(AttnRes)**를 제안했다. [논문 원문](https://arxiv.org/abs/2603.15031)

![기본 residual connection, Full AttnRes, Block AttnRes의 구조 비교]({{ '/assets/images/blog/attention-residuals/architecture.png' | relative_url }})

그림의 왼쪽은 이전 결과를 하나의 residual stream에 계속 더한다. 가운데는 이전 층 출력을 각각 보관하고, 오른쪽은 여러 층의 출력을 블록별 합으로 압축해 참조한다. 이 글의 그림, 의사 코드, 표는 Kimi 팀의 2026년 3월 기술 보고서 v1에서 가져왔다. [Figure 1](https://arxiv.org/pdf/2603.15031v1#page=1)

## 1. 층의 깊이 방향으로 attention을 적용한다

한 토큰 위치를 기준으로 층 $l$의 입력을 $h_l$, 각 층의 변환 출력을 $v_i=f_i(h_i)$, 토큰 임베딩을 $v_0$이라고 하자. 일반 residual connection을 펼치면 $h_l=\sum_{i=0}^{l-1}v_i$다. 이미 섞인 벡터만 다음 층으로 전달되므로, 다음 층에서 특정 출력을 골라 가중치를 다시 줄 수 없다.

**Full AttnRes**는 같은 토큰 위치의 임베딩과 이전 층 출력에 attention을 적용한다. 예를 들어 Transformer의 self-attention에서 `cat` 위치는 같은 문장 안의 여러 토큰을 참조한다. AttnRes에서 `cat` 위치가 참조하는 대상은 그 위치의 임베딩과 하위 층 표현들이다. 논문에서는 self-attention과 MLP를 각각 하나의 층으로 센다.

$$
s_{i,l}=w_l^\top\operatorname{RMSNorm}(v_i),\qquad
\alpha_{i\to l}=\frac{\exp(s_{i,l})}{\sum_{j=0}^{l-1}\exp(s_{j,l})},\qquad
h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i
$$

쿼리 $w_l$은 층마다 하나씩 학습하는 벡터다. 층이 같으면 쿼리를 공유하되, key가 토큰마다 달라 가중치도 달라진다. RMSNorm은 출력의 크기만으로 점수가 커지는 현상을 줄이기 위해 key에 적용한다. 가중합에 들어가는 value는 원래 출력 $v_i$다. [논문 §3.1](https://arxiv.org/pdf/2603.15031v1#page=4)

## 2. Block AttnRes는 여러 층을 하나로 묶는다

Full AttnRes는 토큰마다 이전 출력 $O(Ld)$를 보관하고, 전체 깊이에 걸쳐 $O(L^2d)$의 가중합과 점수 계산을 수행한다. $L$은 층 수, $d$는 은닉 차원이다. 활성값을 재계산하는 학습에서는 원래 버릴 수 있던 출력들을 보관해야 하고, 파이프라인 병렬 학습에서는 다음 스테이지에 이 출력들을 전달해야 한다.

**Block AttnRes**는 $L$개 층을 $N$개 블록으로 나누고, 블록이 끝나면 내부 층 출력의 합을 하나의 벡터로 저장한다. 각 층은 **토큰 임베딩, 완료된 블록의 합, 현재 블록의 부분합**을 가중합한다. 새 블록의 첫 층에는 현재 블록의 부분합이 아직 없다. 이전 표현의 보관 및 전달 비용은 토큰당 $O(Ld)$에서 $O(Nd)$로 줄지만, 완료된 블록 안의 각 층 출력에는 개별 가중치를 줄 수 없어진다. [논문 §3.2](https://arxiv.org/pdf/2603.15031v1#page=5)

![완료된 블록과 현재 부분합을 받아 깊이 방향 가중합을 수행하는 PyTorch 의사 코드]({{ '/assets/images/blog/attention-residuals/pseudocode.png' | relative_url }})

코드에서 `blocks`는 임베딩과 완료된 블록의 합을 들고 있고 `partial_block`은 현재까지의 부분합이다. `stack`으로 참조 대상 축을 만든 뒤 `softmax(0)`으로 그 축을 정규화한다. Self-attention과 MLP는 각자 다른 쿼리로 입력을 만들고, 각각의 출력을 부분합에 더한다. [Figure 2](https://arxiv.org/pdf/2603.15031v1#page=5)

## 3. 학습에서는 전송을, 추론에서는 반복 읽기를 줄인다

![네 개의 물리 rank가 두 가상 스테이지를 처리하며 새 블록만 전달하는 파이프라인]({{ '/assets/images/blog/attention-residuals/pipeline-cache.png' | relative_url }})

**학습.** 파이프라인 병렬 학습에서 하나의 물리 rank가 여러 가상 스테이지를 맡으면, 앞선 가상 스테이지에서 받았던 블록을 같은 rank에 남겨 둘 수 있다. 그림에서 두 번째 가상 스테이지로 넘어갈 때 `+[b₁, b₂]`는 수신 측 캐시에 아직 없는 블록을 뜻한다. 마이크로배치마다 이미 받은 표현을 재사용해 통신량을 줄인다. [Figure 3](https://arxiv.org/pdf/2603.15031v1#page=6)

![완료된 블록의 값을 병렬로 읽고 현재 부분합과 online softmax로 합치는 추론 알고리즘]({{ '/assets/images/blog/attention-residuals/two-phase-inference.png' | relative_url }})

**추론.** 쿼리 벡터는 층의 순전파 결과를 기다릴 필요가 없다. 그래서 첫 단계에서는 한 블록 안의 쿼리를 모아, 이미 완료된 블록들에 대한 점수와 가중합을 한 번에 계산한다. 두 번째 단계에서는 층을 순서대로 실행하면서 그때까지 쌓인 부분합을 후보에 추가한다. 두 후보 집합이 하나의 softmax 분모를 공유하도록, 저장해 둔 최댓값과 지수합을 이용해 결과를 합친다. 이 online softmax 덕분에 이전 블록을 층마다 다시 읽을 필요가 줄어든다. [Algorithm 1](https://arxiv.org/pdf/2603.15031v1#page=7)

![일반 residual, mHC, Full AttnRes, Block AttnRes의 토큰당 층당 residual 메모리 I/O 비교]({{ '/assets/images/blog/attention-residuals/memory-io.png' | relative_url }})

표의 예시 설정($L=128$, $N=8$)에서 토큰당·층당 residual 처리의 총 메모리 I/O는 일반 residual이 $3d$, Full AttnRes가 $24d$, Block AttnRes가 $5.5d$다. 4개 stream을 쓰는 mHC는 약 $34d$다. 이 수치는 층 내부 연산을 제외한 읽기·쓰기 비용이다. 논문은 별도로 자체 추론 워크로드에서 종단 간 지연 시간이 2% 미만 늘었다고 보고했다. [논문 §4.2 및 Table 1](https://arxiv.org/pdf/2603.15031v1#page=8)

## 4. 같은 연산량에서 검증 손실이 낮아졌다

![학습 연산량에 따른 기준 모델, Full AttnRes, Block AttnRes의 검증 손실]({{ '/assets/images/blog/attention-residuals/scaling.png' | relative_url }})

스케일링 실험의 다섯 규모에서 Full과 Block 모두 기준 모델보다 검증 손실이 낮았다. 그림의 가로 화살표는 같은 손실에 도달하는 데 필요한 연산량 차이를 표시한다. 논문의 추정에 따르면 **Block AttnRes와 같은 검증 손실에 도달하려면 기준 모델에 약 1.25배의 학습 연산량**이 필요했다. 가장 큰 스케일링 실험에서 Full과 Block의 손실 차이는 0.001이었다. [Figure 4 및 Table 2](https://arxiv.org/pdf/2603.15031v1#page=9)

![Kimi Linear 실험의 검증 손실, 블록별 출력 크기, 블록별 기울기 크기 비교]({{ '/assets/images/blog/attention-residuals/training.png' | relative_url }})

별도로 총 48B, 토큰당 활성 3B 파라미터인 Kimi Linear에 Block AttnRes를 적용했다. 사전학습 1T 토큰과 중간 학습 약 400B 토큰을 사용했다. 첫 1T 토큰 학습을 비교한 위 그림에서는 깊은 Transformer 블록의 출력 크기가 덜 커졌고, 앞쪽에 집중되던 기울기 크기가 깊이 방향으로 더 고르게 분포했다. [Figure 5](https://arxiv.org/pdf/2603.15031v1#page=10)

![기준 모델과 Block AttnRes의 일반 지식, 수학, 코딩, 중국어 벤치마크 점수 비교]({{ '/assets/images/blog/attention-residuals/benchmarks.png' | relative_url }})

최종 벤치마크 15개 중 14개에서 점수가 올랐다. GPQA-Diamond는 **36.9 → 44.4**, Math는 **53.5 → 57.1**, HumanEval은 **59.1 → 62.2**였다. MMLU-Pro는 양쪽 모두 **52.2**였다. [Table 3](https://arxiv.org/pdf/2603.15031v1#page=10)

## 5. 모델의 깊이와 실제 참조 패턴

![같은 학습 연산량에서 모델 폭과 깊이의 비율을 달리한 25개 구성의 검증 손실]({{ '/assets/images/blog/attention-residuals/architecture-sweep.png' | relative_url }})

연구진은 학습 연산량을 고정하고 모델 폭 $d_{\mathrm{model}}$과 attention head 수 $H$를 Transformer 블록 수 $L_b$에 대한 비율로 바꿨다. 기준 모델의 최저 손실은 $d_{\mathrm{model}}/L_b=60$에서 1.847이었고, AttnRes의 최저 손실은 비율 45에서 1.802였다. 이 범위에서는 AttnRes를 적용했을 때 폭보다 깊이에 더 비중을 둔 구성이 유리했다. 깊은 모델은 순차적으로 실행할 층이 많아지므로, 배포할 모델을 고를 때는 지연 시간도 함께 봐야 한다. [Figure 7](https://arxiv.org/pdf/2603.15031v1#page=12)

![Full과 Block AttnRes가 self-attention 및 MLP 입력에 사용하는 깊이 방향 가중치 히트맵]({{ '/assets/images/blog/attention-residuals/depth-weights.png' | relative_url }})

히트맵의 각 행은 한 층이 이전 표현에 준 가중치를 토큰 전체에서 평균한 값이다. 대각선 근처가 짙어 대부분의 층이 가까운 출력을 많이 참조한다. 동시에 첫 번째 열의 임베딩에 지속적으로 가중치가 남고, 일부 층에서는 멀리 떨어진 출력에도 가중치가 몰린다. 그림 왼쪽의 self-attention 입력은 오른쪽의 MLP 입력보다 참조를 더 넓게 분산하는 경향을 보인다. [Figure 8](https://arxiv.org/pdf/2603.15031v1#page=13)

## 정리

AttnRes는 각 층이 이전 표현을 고정된 합으로 받는 대신, 토큰마다 다시 가중합하도록 residual 경로를 바꾼다. Full AttnRes는 개별 층까지 선택할 수 있고, Block AttnRes는 완료된 블록을 하나의 벡터로 압축해 저장과 통신 비용을 줄인다. 논문의 실험에서는 블록 단위로 압축하고도 Full AttnRes의 검증 손실 개선을 상당 부분 유지했다.

[논문 PDF](https://arxiv.org/pdf/2603.15031v1) · [공식 저장소](https://github.com/MoonshotAI/Attention-Residuals)
