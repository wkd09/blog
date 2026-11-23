---
title: "Inkling 2 Mini: 같은 레이어를 세 번 돌리면 정보는 어떻게 흐를까?"
date: 2026-11-24 00:00:00 +0900
summary: "Goodfire의 분석을 바탕으로 Inkling 2 Mini가 이전 블록과 이전 루프의 출력을 어떻게 읽는지 살펴본다."
categories:
  - research
tags:
  - AI
  - LLM
  - Transformer
  - Attention Residuals
  - Interpretability
source: "Goodfire — Interpreting Inkling 2 Mini"
---

Thinking Machines Lab이 로컬에서도 실행할 수 있는 공개 가중치 모델 **Inkling 2 Mini**를 내놓았다. 지난 글에서 정리한 Attention Residuals를 사용하고, 중간 레이어를 같은 가중치로 여러 번 실행한다. 저는 이 두 가지가 같이 쓰일 때 다음 루프가 이전 루프에서 무엇을 읽는지가 궁금했다. [Thinking Machines Lab의 모델 발표](https://thinkingmachines.ai/news/inkling-2-mini/)

Goodfire는 세 가지 질문을 던졌다. 각 레이어의 입력은 어디에서 오는가? 어떤 레이어가 블록과 루프 사이에서 정보를 주고받는가? 루프 중간의 표현으로 최종 출력을 얼마나 잘 읽어낼 수 있는가? 그림과 수식 다섯 개를 따라가면 앞의 두 질문부터 감을 잡을 수 있다. [Goodfire의 분석](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

## Inkling 2 Mini의 구조

24개 레이어는 6개씩 네 블록으로 묶인다. 각 레이어에서는 attention 하위 레이어가 먼저 실행되고, 이어서 MoE(Mixture of Experts) 하위 레이어가 실행된다.

![입력 블록 B1, 세 차례 반복되는 B2와 B3, 출력 블록 B4를 잇는 전체 구조]({{ '/assets/images/blog/inkling-2-mini/three-loop-architecture.png' | relative_url }})

전체 구조를 그린 Goodfire의 그림 3. 왼쪽 아래의 입력에서 시작해 오른쪽 위의 출력으로 올라간다.

### 깊이 방향 어텐션

{% capture attention_residuals_url %}{% post_url 2026-11-23-attention-residuals %}{% endcapture %}
지난 글에서 다룬 Block Attention Residuals에서는 각 블록이 자체적으로 하위 레이어 출력을 누적한다. 새 하위 레이어는 토큰 임베딩, 이미 계산이 끝난 블록의 합, 현재 블록에서 지금까지 계산한 부분합을 읽고 입력에 따라 각각의 가중치를 정한다. 그래서 바로 앞 블록을 지나온 뒤에도 더 앞쪽 블록의 출력을 직접 읽을 수 있다. [이전 글에서 정리한 Attention Residuals]({{ attention_residuals_url | relative_url }})

![16번 레이어의 attention 입력에 임베딩, B1과 B2의 합, B3의 13~15번 레이어 부분합이 들어가는 그림]({{ '/assets/images/blog/inkling-2-mini/depth-wise-attention.png' | relative_url }})

예를 들어 16번 레이어의 attention이 실행될 때 B3에서 읽을 수 있는 값은 13~15번 레이어의 부분합이다. 바로 뒤의 MoE가 입력을 만들 때는 16번 레이어의 attention 출력도 그 부분합에 들어간다. 그림은 Goodfire의 그림 1이다.

### 중간 레이어를 세 번 실행한다

B1(1~6번 레이어)이 입력을 한 번 처리하면, B2와 B3(7~18번 레이어)를 묶어 같은 가중치로 세 번 실행한다. 마지막으로 B4(19~24번 레이어)가 한 번 실행돼 출력을 만드는 데 쓰인다. 가중치를 가진 레이어는 24개이고, 토큰 하나를 처리할 때 레이어를 실행하는 횟수는 $6 + 3 \times 12 + 6 = 48$번이다. [Thinking Machines Lab의 루프 설계 설명](https://thinkingmachines.ai/news/inkling-2-mini/#looping-design)

여기서 처음 본 게 **크로스 루프 어텐션(cross-loop attention)**이다. 코어 블록의 두 번째와 세 번째 실행에서는 각 레이어가 직전 루프에서 끝까지 계산한 B2와 B3의 출력을 읽는다. 구체적으로는 attention의 Q·K·V와 MoE 입력이 각각의 가중치로 두 블록을 섞어 메시지를 만든다. [Goodfire의 아키텍처 부록](https://www.goodfire.com/research/interpreting-inkling-2-mini/#architecture-details)

![두 번째 루프의 16번 레이어가 깊이 방향으로 읽는 파란 경로와 첫 번째 루프를 읽는 분홍 경로]({{ '/assets/images/blog/inkling-2-mini/layer-inputs-blocks.jpg' | relative_url }})

Goodfire의 그림 2에서 파란 선은 임베딩과 앞서 계산된 블록·부분합을 읽는 경로다. 분홍 선은 직전 루프의 B2와 B3를 읽는 경로다. 그래서 두 번째 루프의 16번 레이어가 첫 번째 루프의 18번 레이어 출력을 간접적으로 읽을 수 있다. 두 경로 모두 한 토큰 위치의 벡터를 다룬다. 토큰 위치 사이의 정보 교환은 각 레이어의 시퀀스 어텐션이 담당한다.

## 레이어 입력을 수식 다섯 개로 풀어 보기

![각 레이어가 입력을 읽을 때 깊이 방향과 직전 루프의 기여를 순서대로 강조하는 애니메이션]({{ '/assets/images/blog/inkling-2-mini/information-flow.gif' | relative_url }})

Goodfire의 그림 4는 레이어에 들어오는 벡터를 여러 출력의 합으로 풀어서 보여 준다. 저는 먼저 블록 단위로 보고, 그다음 블록을 이루는 하위 레이어 단위로 내려가는 편이 이해하기 쉬웠다.

### 1. 깊이 방향으로 입력 만들기

$$
\mathbf{x}=\sum_i\alpha_i r_i\mathbf{v}_i
$$

| 기호 | 의미 |
| --- | --- |
| $\mathbf{x}$ | 깊이 방향 어텐션으로 만든 입력 벡터 |
| $\alpha_i$ | 소프트맥스로 정한 블록 $i$의 어텐션 가중치 |
| $r_i$ | 블록 벡터의 크기를 맞추는 RMS 정규화 계수 $1/\operatorname{RMS}(\mathbf{v}_i)$ |
| $\mathbf{v}_i$ | 읽을 수 있는 블록의 합 벡터. 토큰 임베딩도 별도 항으로 들어간다. |

쉽게 말하면 지금 읽을 수 있는 벡터의 크기를 정규화하고, 각 벡터를 얼마나 읽을지 정해서 더하는 식이다.

### 2. 블록 출력 펼치기

$$
\mathbf{v}_i=\sum_{w\in B_i}\mathbf{o}_w
$$

$\mathbf{o}_w$는 블록 $B_i$에 속한 attention 또는 MoE 하위 레이어 하나의 출력이다. 완료된 블록은 여섯 레이어의 출력을 모두 더하고, 현재 블록의 부분합에는 그 시점까지 나온 출력만 들어간다. 이 식을 첫 번째 식에 대입하면 블록 단위 가중합을 하위 레이어 출력 단위로 펼칠 수 있다.

### 3. 직전 루프에서 메시지 가져오기

$$
\mathbf{m}=\beta_2r_2^{[t-1]}\mathbf{v}_2^{[t-1]}
+\beta_3r_3^{[t-1]}\mathbf{v}_3^{[t-1]}
$$

현재 루프가 $t$라면 $t-1$은 직전 루프다. 직전 루프의 B2와 B3 출력을 각각 정규화하고, 크로스 루프 어텐션 가중치 $\beta_2$와 $\beta_3$를 곱해 메시지 $\mathbf{m}$을 만든다. 세 번째 루프에서는 두 번째 루프의 두 블록을 읽는다.

### 4. 현재 입력과 이전 메시지 섞기

$$
\mathbf{x}'=\boldsymbol{\kappa}\odot\mathbf{x}
+\boldsymbol{\gamma}\odot\mathbf{m}
$$

$\odot$는 벡터 원소별 곱이다. $\boldsymbol{\kappa}$는 현재 입력의 각 성분을 줄이거나 키운다. $\boldsymbol{\gamma}$는 직전 루프 메시지의 각 성분을 얼마나 더하거나 뺄지 정한다. 두 계수를 적용한 벡터를 더하면 하위 레이어에 전달할 입력이 만들어진다.

### 5. 각 출력의 기여도 계산하기

$$
s_w=\frac{\mathbf{c}_w\cdot\mathbf{x}'}{\|\mathbf{x}'\|^2}
$$

$\mathbf{c}_w$는 이전 출력 하나에 해당 경로의 어텐션 가중치, 정규화 계수, 게이트를 적용한 벡터다. 이런 항을 모두 더하면 $\mathbf{x}^{\prime}$가 된다. 위 식은 각 항이 최종 입력과 같은 방향으로 얼마나 더해졌는지 계산한다. 반대 방향이면 기여도가 음수가 될 수 있고, 전체 항의 기여도를 더하면 1이다. [Goodfire의 입력 분해 설명](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

## 실제로 어느 레이어를 많이 읽을까?

Goodfire는 FineWeb 문서 400개에서 가져온 13,233개 토큰의 기여도를 평균했다. 깊이 방향에서는 대체로 가까운 레이어의 출력이 더 크게 반영됐다. 슬라이딩 윈도 어텐션이 가까운 토큰을 주로 보는 것과 비슷한 경향이 **레이어 깊이**에서 나타난 셈이다. 더 앞쪽 블록도 여전히 읽을 수 있다. 크로스 루프에서는 직전 루프의 B3를 B2보다 많이 읽는 경향이 있었고, 특히 B3가 직전 루프의 B3를 읽는 패턴이 뚜렷했다. [Goodfire의 입력 기여도 분석](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

연구진은 이 관찰을 바탕으로 가중치도 들여다봤다. 아래 두 그래프는 MoE 뉴런의 출력 방향(writer)과 블록을 고르는 어텐션 쿼리 방향(reader) 사이의 절대 코사인 유사도가 0.30 이상인 쌍을 센 것이다. 가로축은 쓰는 레이어, 막대의 색은 읽는 레이어, 높이는 조건을 만족한 쌍의 개수다.

![깊이 방향 어텐션의 writer–reader 쌍이 12번과 18번 레이어에 두드러지게 모인 그래프]({{ '/assets/images/blog/inkling-2-mini/reader-depthmix.jpg' | relative_url }})

깊이 방향에서는 블록 끝인 6·12·18번 레이어가 눈에 띈다. 특히 12번 레이어는 B3와 B4의 여러 reader와 높은 정렬을 보인다. Goodfire의 그림 5다.

![크로스 루프 어텐션의 writer–reader 쌍이 18번 레이어에 집중되고 13번 reader가 가장 큰 몫을 차지하는 그래프]({{ '/assets/images/blog/inkling-2-mini/reader-crossloop.jpg' | relative_url }})

크로스 루프에서는 B3의 마지막인 18번 레이어와 다음 루프에서 B3를 시작하는 13번 레이어의 연결이 특히 두드러진다. 연구진은 블록 끝 레이어가 이후 레이어에서 해당 블록을 찾기 쉽도록 활성화에 일종의 표시를 남길 수 있다고 해석했다. Goodfire의 그림 6이다. [가중치 공간 분석](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

## 루프 중간에서 다음 토큰을 읽을 수 있을까?

Goodfire는 중간 상태에 최종 출력 헤드를 곧바로 적용하는 logit lens도 시험했다. 각 루프의 끝부분에서는 어느 정도 작동했지만, 그보다 앞선 레이어에서는 최종 출력을 잘 복원하지 못했다. 중간 표현을 학습된 변환으로 보정하는 tuned lens와 마지막 블록 B4를 적용하는 coda lens는 두 평가 지표에서 전반적으로 더 잘 작동했다. [Goodfire의 루프 디코딩 실험](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

## 마치며

요즘 로컬에서 돌릴 수 있는 소형 모델이 빠르게 좋아지고 있다. 이번 모델에서 특히 재밌었던 건 같은 레이어가 세 번 실행되면서, 앞선 루프의 자기 블록에서 정보를 가져온다는 점이다. 13번 레이어가 앞선 루프의 18번 레이어 출력을 담은 블록을 읽는 예를 보고 나니, 중간 레이어를 반복한다는 말이 조금 더 구체적으로 와닿았다. 앞으로 다른 루프 모델에서도 블록 끝과 다음 루프의 시작 사이에 비슷한 패턴이 보일지 궁금하다.
