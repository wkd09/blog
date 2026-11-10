---
title: "RoFormer 논문 정리: 위치 정보를 회전으로 표현하는 RoPE"
date: 2026-07-17 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - NLP
  - LLM
  - Transformer
  - PositionalEncoding
  - RoPE
  - Paper
source: "arXiv:2104.09864"
---

# RoFormer: 위치 정보를 회전으로 표현하는 RoPE

이 글은 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)을 바탕으로 정리한 글이다.

> Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu  
> [[Paper](https://arxiv.org/pdf/2104.09864)]

핵심 관심사는 하나다.

> 토큰의 절대 위치를 query와 key의 회전으로 표현하면, self-attention의 내적 안에 상대적인 위치 차이를 자연스럽게 넣을 수 있다.

이 방법이 `Rotary Position Embedding`, 줄여서 **RoPE**다. RoFormer는 RoPE를 적용한 Transformer encoder 모델의 이름이다. 오늘날 LLaMA 계열을 비롯한 여러 LLM이 RoPE를 사용하는 이유도 이 구조가 위치 표현과 attention 계산을 깔끔하게 결합하기 때문이다.

## 1. 왜 위치 정보가 필요한가

Transformer의 self-attention은 토큰 사이의 관계를 직접 계산한다. 하지만 순서 자체를 이해하는 장치는 기본적으로 없다.

```text
나는 학교에 갔다
학교에 나는 갔다
```

두 문장에 같은 단어가 들어 있어도 순서가 다르면 의미가 달라진다. 따라서 Transformer에는 각 토큰이 몇 번째에 있는지 알려주는 positional encoding이 필요하다.

기존 방식은 크게 두 가지였다.

| 방식 | 아이디어 | 한계 |
|---|---|---|
| Absolute positional embedding | 각 위치에 학습 가능한 벡터를 더함 | 학습 때 보지 못한 긴 위치에 약할 수 있음 |
| Sinusoidal encoding | 사인·코사인 함수로 위치 벡터를 계산함 | 위치 벡터가 attention의 상대 거리와 직접 연결되지는 않음 |

RoPE는 위치 벡터를 hidden state에 단순히 더하지 않는다. query와 key를 위치에 따라 회전시킨다.

## 2. RoPE의 핵심 아이디어

2차원 벡터를 각도 $m\theta$만큼 회전하는 행렬은 다음과 같다.

$$
R_{m\theta} =
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta)\\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
$$

위치 $m$에 있는 query와 key를 각각 회전하면 다음과 같이 쓸 수 있다.

$$
q_m' = R_{m\theta}q_m, \qquad k_n' = R_{n\theta}k_n
$$

여러 차원의 hidden state에서는 차원을 2개씩 묶어 각기 다른 주파수로 회전시킨다. 낮은 차원은 빠르게 회전하고, 높은 차원은 천천히 회전한다. 덕분에 하나의 벡터 안에 짧은 거리와 긴 거리에 대한 여러 주기의 위치 정보가 함께 들어간다.

## 3. 회전했는데 왜 상대 위치가 되는가

RoPE에서 가장 중요한 성질은 두 회전된 벡터의 내적을 전개하면 드러난다.

$$
\langle R_{m\theta}q, R_{n\theta}k\rangle
= \langle q, R_{(n-m)\theta}k\rangle
$$

왼쪽에는 위치 $m$, $n$이 각각 들어가 있지만, 오른쪽에는 위치의 차이인 $n-m$만 남는다. 즉 query와 key를 절대 위치에 따라 회전시켰지만, attention score는 상대적인 거리 정보를 반영하게 된다.

직관적으로 말하면 다음과 같다.

```text
토큰 m의 query를 m만큼 회전
토큰 n의 key를 n만큼 회전
두 벡터의 각도 차이 = (n - m)만큼의 상대 위치
```

이것이 RoPE의 매력이다. 별도의 relative position bias를 attention score에 추가하지 않아도, query-key 내적 자체가 상대 위치를 표현한다.

## 4. 복소수 관점의 간단한 이해

2차원 회전은 복소수 곱셈으로도 표현할 수 있다. 두 차원을 하나의 복소수로 보고 위치 $m$에서 다음과 같이 곱한다.

$$
q_m' = q_m e^{im\theta}, \qquad k_n' = k_n e^{in\theta}
$$

query와 key가 서로 다른 위치에 있으므로, attention은 두 토큰 사이의 위상 차이를 보게 된다. 이 위상 차이가 상대 위치 $n-m$에 해당한다.

실제 구현에서는 복소수 연산을 직접 수행하기보다, hidden dimension을 짝수 차원 쌍으로 나누고 사인·코사인 값을 이용해 회전한다.

## 5. RoPE의 세 가지 장점

### 5.1 절대 위치와 상대 위치를 동시에 얻는다

각 토큰은 자신의 위치에 따라 회전하므로 절대 위치 정보가 표현된다. 동시에 attention score에는 위치 차이가 나타난다. 두 종류의 positional information을 별도의 복잡한 모듈 없이 하나의 연산으로 처리한다.

### 5.2 거리가 멀어질수록 의존성이 약해지는 경향

RoFormer는 서로 다른 주파수의 회전을 여러 차원에 적용한다. 여러 주파수 성분을 합치면 상대 거리가 커질수록 위치 정보의 상관이 평균적으로 감소한다.

논문은 이를 inter-token dependency가 거리에 따라 감쇠하는 성질로 분석한다. 아래 그래프처럼 상대 거리가 커질수록 attention에서 위치 정보가 제공하는 상관이 전반적으로 낮아지는 경향을 보인다.

![RoPE relative distance]({{ '/assets/images/blog/roformer-relative-distance.png' | relative_url }})

*상대 거리가 증가할수록 위치 기반 의존성이 전반적으로 감쇠한다. 세로축은 논문의 relative upper bound를 옮긴 것이다.*

다만 이것은 모든 attention weight가 무조건 거리에 따라 작아진다는 뜻은 아니다. 내용 정보와 학습된 query·key 값의 영향은 여전히 남아 있다. RoPE가 제공하는 위치 성분에 대한 이론적 경향으로 이해하는 편이 정확하다.

### 5.3 길이 일반화에 유리하다

학습 가능한 absolute embedding은 학습 시 정해진 최대 위치 개수에 묶이기 쉽다. 반면 RoPE는 위치에 따라 사인·코사인 회전값을 계산하므로, 원리상 새로운 위치에도 적용할 수 있다.

물론 학습 길이를 크게 넘어선 context에서 성능이 자동으로 보장되는 것은 아니다. 실제 LLM에서는 context length, base frequency, scaling 방법에 따라 extrapolation 성능이 달라진다. 이후에는 position interpolation, NTK-aware scaling, YaRN 같은 확장 기법이 이 문제를 추가로 다룬다.

## 6. RoPE를 attention에 넣는 위치

일반적인 self-attention은 다음과 같다.

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

RoPE는 value $V$가 아니라 query $Q$와 key $K$에 적용한다.

```text
Q = XW_Q
K = XW_K
V = XW_V

Q_rope = RoPE(Q, position)
K_rope = RoPE(K, position)

Attention = softmax(Q_rope K_rope^T / sqrt(d)) V
```

value까지 회전시키지 않는 이유는 위치 정보가 attention의 어느 토큰을 얼마나 볼지 결정하는 score에 들어가면 충분하기 때문이다. 실제 구현에서도 보통 query와 key projection 이후, attention score를 계산하기 전에 RoPE를 적용한다.

## 7. RoFormer 실험 결과

논문은 긴 텍스트 분류 벤치마크에서 RoFormer를 평가하고 Transformer-base, BERT 등과 비교한다.

먼저 Transformer-base와 비교한 WMT 영어-독일어 번역 결과에서 RoFormer가 BLEU 27.5로 Transformer-base의 27.3보다 높았다.

![RoFormer BLEU result]({{ '/assets/images/blog/roformer-bleu.png' | relative_url }})

*Transformer-base 대비 RoFormer의 BLEU 비교. 작은 차이처럼 보여도 positional encoding만 바꾼 효과를 확인하는 결과다.*

문장 수준 분류 결과는 데이터셋마다 강점이 달랐다.

![RoFormer sentence classification results]({{ '/assets/images/blog/roformer-sentence-results.png' | relative_url }})

| 모델 | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI (m/mm) |
|---|---:|---:|---:|---:|---:|---:|
| BERT | 88.9 | **93.5** | **90.5** | 85.8 | 71.2 | **84.6/83.4** |
| RoFormer | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

RoFormer가 모든 과제에서 이기는 것은 아니다. 하지만 MRPC, STS-B, QQP처럼 문장 간 관계를 보는 과제에서는 개선이 나타났고, positional encoding의 선택이 downstream task에 영향을 줄 수 있음을 보여준다.

논문은 긴 텍스트 분류에서도 RoPE의 장점을 확인한다. 특히 최대 길이 512와 1024를 비교한 결과, RoFormer-1024는 validation 66.07%, test 69.79%를 기록했다.

![RoFormer long text classification results]({{ '/assets/images/blog/roformer-long-text-results.png' | relative_url }})

*긴 텍스트 분류 결과. RoFormer-1024가 더 긴 입력 길이를 사용했을 때 validation과 test 모두에서 가장 높은 결과를 보인다.*

학습 과정에서 RoPE를 적용한 모델의 loss도 비교적 안정적으로 감소한다.

![RoFormer training loss]({{ '/assets/images/blog/roformer-loss.png' | relative_url }})

*RoFormer와 BERT의 학습 loss 비교, 그리고 RoPE 적용 여부에 따른 Performer의 loss 비교.*

## 8. Linear attention에도 적용할 수 있다

RoFormer 논문이 흥미로운 이유는 RoPE를 일반적인 softmax attention에만 묶어 두지 않았기 때문이다. 저자들은 RoPE의 상대 위치 표현을 linear attention에도 결합할 수 있다고 설명한다.

일반적인 linear attention은 kernel feature map을 이용해 attention의 계산 순서를 바꾼다.

$$
\operatorname{Attention}(Q,K,V)
\approx \phi(Q)\left(\phi(K)^T V\right)
$$

RoPE는 query와 key에 회전을 적용하므로, 회전 행렬을 feature map 내부에 포함하는 방식으로 상대 위치 정보를 전달할 수 있다. 따라서 RoPE는 특정 attention 구현의 부가 기능이라기보다, query-key 관계에 위치 구조를 넣는 일반적인 방법으로 볼 수 있다.

## 9. RoPE의 한계

RoPE가 강력해도 만능은 아니다.

- 학습한 context length를 훨씬 넘어가면 회전 주기가 예상과 다르게 사용되어 성능이 떨어질 수 있다.
- 긴 context에서 position index가 커질수록 주파수별 위상이 빠르게 변해 extrapolation이 불안정해질 수 있다.
- 상대 거리에 따른 감쇠 성질은 위치 성분의 경향이지, 모든 attention이 가까운 토큰만 보도록 강제하는 규칙은 아니다.
- 최근 LLM에서는 단순히 RoPE를 사용하는 것보다 base frequency와 scaling 전략을 모델의 학습 길이와 함께 설계하는 것이 중요하다.

즉 RoPE의 “길이 유연성”은 새로운 길이에 계산할 수 있다는 뜻이지, 아무런 추가 학습이나 조정 없이 무한히 잘 동작한다는 뜻은 아니다.

## 10. 현재 LLM에서 RoPE가 중요한 이유

RoPE는 2021년에 발표된 positional encoding 연구지만, 이후 LLM 아키텍처에 큰 영향을 주었다. LLaMA 계열과 여러 decoder-only Transformer는 RoPE를 기본 위치 표현으로 사용한다.

그 이유는 실용적이다.

```text
추가 positional embedding table이 필요하지 않다
query와 key에 적용하는 연산이 단순하다
attention score에 상대 위치 정보가 자연스럽게 들어간다
context length 확장 연구와 연결하기 쉽다
```

특히 긴 문서를 처리하는 RAG나 긴 대화형 Agent에서는 “현재 토큰과 참조 토큰이 얼마나 떨어져 있는가”가 중요하다. RoPE는 이 거리 정보를 attention 계산에 직접 반영할 수 있는 기반을 제공한다. 다만 긴 context를 실제로 안정적으로 사용하려면 KV cache, attention 효율화, position scaling을 함께 고려해야 한다.

## 내가 이해한 핵심

RoPE를 처음 보면 “position embedding을 회전으로 바꾼 방법”처럼 보인다. 하지만 내가 이해한 핵심은 회전 자체보다 query-key 내적에 상대 위치 차이가 남는다는 점이다.

> query와 key를 각자의 위치만큼 회전하면, 두 벡터의 내적에는 두 위치의 차이가 남는다.

이 한 가지 성질이 RoPE의 대부분을 설명한다.

1. 각 토큰은 자신의 위치에 따라 회전하므로 절대 위치가 표현된다.
2. query-key 내적에서는 상대 위치 차이가 나타난다.
3. 여러 주파수를 사용하면 거리 변화에 따른 위치 상관을 표현할 수 있다.
4. 별도의 큰 embedding table 없이 attention의 기존 흐름에 넣을 수 있다.

결국 RoPE는 positional encoding을 attention 바깥에 덧붙이는 방식이 아니라, **attention이 두 토큰의 관계를 계산하는 좌표계 자체를 위치에 따라 회전시키는 방법**이다. 오늘날 LLM에서 RoPE를 이해하는 것은 context length 확장과 long-context 성능을 이해하기 위한 출발점이기도 하다.

## 참고 자료

- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), arXiv:2104.09864.
- 논문 결과 이미지는 위 논문의 실험 결과를 바탕으로 정리한 것이다.
