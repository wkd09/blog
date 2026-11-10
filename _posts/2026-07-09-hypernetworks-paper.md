---
title: "HyperNetworks 논문 정리: 가중치를 생성하는 네트워크"
date: 2026-07-09 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - DeepLearning
  - RNN
  - LSTM
  - HyperNetwork
  - Paper
source: "arXiv:1609.09106"
---

# HyperNetworks

이 글은 논문 [HyperNetworks](https://arxiv.org/abs/1609.09106)를 바탕으로 정리한 글이다.

> David Ha, Andrew Dai, Quoc V. Le  
> Google Brain. [[Paper](https://arxiv.org/pdf/1609.09106)]

핵심 관심사는 하나다.

> 하나의 네트워크가 다른 네트워크의 weight를 직접 학습하는 대신, 그 weight를 만들어내는 함수를 학습하게 하자.

일반적인 neural network에서는 weight 자체가 학습 대상이다. 하지만 HyperNetwork에서는 작은 네트워크가 main network의 weight를 생성한다. 논문은 이 관계를 genotype과 phenotype에 비유한다.

```text
HyperNetwork:
weight를 생성하는 작은 네트워크

Main Network:
생성된 weight로 실제 task를 수행하는 네트워크
```

한 줄로 요약하면 이렇다.

> HyperNetwork는 weight 자체를 직접 저장하는 대신, weight를 생성하는 network를 학습해서 parameter sharing과 model expressiveness 사이의 중간 지점을 만든다.

## 1. 왜 HyperNetwork가 필요한가

논문은 CNN과 RNN을 하나의 스펙트럼으로 본다.

RNN은 모든 time step에서 같은 weight를 공유한다.

```text
h_t = f(W h_{t-1}, U x_t)
```

여기서 `W`, `U`는 시퀀스 전체에서 고정되어 있다. 이 방식은 parameter efficient하지만, 모든 time step에 같은 변환을 강제한다는 점에서 유연성이 떨어질 수 있다.

반대로 CNN, 특히 deep convolutional network는 layer마다 서로 다른 weight를 가진다. 표현력은 높지만 깊어질수록 parameter가 많아진다.

```text
RNN:
강한 weight sharing
parameter는 적지만 표현 제약이 큼

CNN:
layer별 non-shared weight
표현력은 크지만 parameter가 많음
```

HyperNetwork는 이 둘 사이의 절충안이다.

> 완전히 같은 weight를 공유하지도 않고, 모든 layer/time step마다 독립 weight를 직접 저장하지도 않는다. 대신 작은 network가 상황에 맞는 weight를 생성한다.

이 관점이 논문의 가장 중요한 출발점이다.

## 2. HyperNetwork의 기본 아이디어

일반적인 network가 다음처럼 weight를 직접 가진다고 하자.

$$
y = f(x; W)
$$

HyperNetwork에서는 weight `W`를 직접 parameter로 두지 않고, 다른 network `g`가 생성한다.

$$
W = g(z)
$$

그래서 main network는 다음처럼 동작한다.

$$
y = f(x; g(z))
$$

여기서 `z`는 layer embedding일 수도 있고, recurrent model에서는 현재 input과 hidden state를 반영한 dynamic embedding일 수도 있다.

이 구조의 의미는 꽤 크다.

```text
기존 방식:
weight 자체를 학습한다.

HyperNetwork 방식:
weight를 만들어내는 규칙을 학습한다.
```

즉 모델은 단순히 parameter 값을 외우는 것이 아니라, parameter를 생성하는 더 작은 함수를 학습한다.

## 3. Static HyperNetwork: CNN weight 생성

논문에서 Static HyperNetwork는 convolutional network의 kernel을 생성하는 데 사용된다.

각 layer마다 embedding vector `z_j`가 있고, hypernetwork는 이 embedding을 입력으로 받아 해당 layer의 kernel을 만든다.

$$
K_j = g(z_j)
$$

여기서 중요한 점은 모든 layer의 kernel이 완전히 독립적으로 학습되는 것이 아니라, 하나의 generator `g`를 공유한다는 것이다.

```text
layer마다 다른 embedding z_j
공통 hypernetwork g
생성된 layer별 kernel K_j
```

이 방식은 deep CNN의 layer들 사이에 부드러운 형태의 weight sharing을 건다. 따라서 parameter 수는 크게 줄일 수 있지만, 각 layer가 완전히 자유로운 kernel을 갖지는 못한다.

논문은 CIFAR-10에서 Wide ResNet 구조에 이 방식을 적용한다. 결과는 다음과 같은 trade-off를 보여준다.

| Model | Test Error | Param Count |
|---|---:|---:|
| Wide Residual Network 40-1 | 6.73% | 0.563M |
| Hyper Residual Network 40-1 | 8.02% | 0.097M |
| Wide Residual Network 40-2 | 5.66% | 2.236M |
| Hyper Residual Network 40-2 | 7.23% | 0.148M |

정확도는 떨어진다. 하지만 parameter 수는 크게 줄어든다.

이 결과를 보면 CNN 쪽에서 HyperNetwork는 성능 향상 기술이라기보다는, **parameter compression과 relaxed weight sharing** 기법에 가깝다.

## 4. Dynamic HyperNetwork: RNN weight를 시간마다 바꾸기

이 논문의 더 흥미로운 부분은 RNN/LSTM에 적용한 Dynamic HyperNetwork다.

기본 RNN은 다음과 같이 time step마다 같은 weight를 쓴다.

$$
h_t = \phi(W_h h_{t-1} + W_x x_t + b)
$$

Dynamic HyperNetwork에서는 `W_h`, `W_x`, `b`가 고정되어 있지 않다. 작은 recurrent hypernetwork가 현재 상태를 보고 이 weight들을 생성하거나 조정한다.

![HyperRNN overview]({{ '/assets/images/blog/hypernetworks-hyperrnn-overview.png' | relative_url }})

*검은색은 main RNN/LSTM, 주황색은 hypernetwork다. Hypernetwork는 현재 input과 hidden state를 보고 main network의 weight 조정 신호를 만든다.*

개념적으로는 다음과 같다.

$$
W_t = g(x_t, h_{t-1}, \hat{h}_{t-1})
$$

$$
h_t = f(x_t, h_{t-1}; W_t)
$$

즉 recurrent model이 time step마다 같은 모델을 반복 적용하는 것이 아니라, 매 순간 조금씩 다른 모델을 사용하게 된다.

논문에서는 이 모델을 HyperRNN이라고 부르고, LSTM에 확장한 버전을 HyperLSTM이라고 부른다.

## 5. 메모리 문제와 weight scaling

가장 직접적인 방법은 hypernetwork가 매 time step마다 전체 weight matrix를 생성하는 것이다.

하지만 이 방식은 너무 비싸다. 큰 LSTM에서 매 step마다 full matrix를 생성하면 memory 사용량이 크게 늘어난다.

그래서 논문은 더 효율적인 방식을 사용한다.

> full weight matrix를 새로 만들지 않고, 기존 weight matrix의 row를 scaling하는 vector를 생성한다.

예를 들어 어떤 weight matrix `W`가 있을 때, hypernetwork는 `d(z)`라는 scaling vector를 만든다.

$$
W(z) =
\begin{bmatrix}
d_0(z) W_0 \\
d_1(z) W_1 \\
\cdots \\
d_n(z) W_n
\end{bmatrix}
$$

이렇게 하면 full matrix를 생성하는 것보다 훨씬 싸다. 하지만 time step마다 main LSTM의 계산 방식은 바뀐다.

직관적으로 보면 HyperLSTM은 이런 일을 한다.

```text
현재 문맥을 본다.
이번 step에서 어떤 gate를 더 강하게 쓸지 정한다.
main LSTM weight의 일부 방향을 키우거나 줄인다.
그 조정된 계산으로 다음 hidden state를 만든다.
```

이 구조는 Layer Normalization과도 닮은 점이 있다. 둘 다 activation으로 들어가기 전의 scale을 조정한다. 하지만 차이가 있다.

```text
Layer Norm:
현재 activation의 평균/분산을 이용한 통계적 정규화

HyperLSTM:
작은 recurrent network가 학습한 동적 scaling policy
```

즉 HyperLSTM은 hand-crafted normalization이 아니라, sequence context에 따라 scale 조정 정책을 직접 학습한다.

## 6. HyperLSTM 구조

LSTM에는 네 개의 gate가 있다.

```text
input gate
candidate gate
forget gate
output gate
```

HyperLSTM은 각 gate에 대해 weight와 bias를 조정하는 embedding을 만든다.

논문에서는 gate 하나를 `y ∈ {i, g, f, o}`라고 두고, 각 gate의 hidden weight, input weight, bias에 대해 embedding을 생성한다.

```text
z_h^y:
hidden-to-hidden weight를 조정하는 embedding

z_x^y:
input-to-hidden weight를 조정하는 embedding

z_b^y:
bias를 조정하는 embedding
```

그리고 이 embedding으로 scaling vector와 bias를 만든다.

$$
d_h^y(z_h) = W_{hz}^y z_h
$$

$$
d_x^y(z_x) = W_{xz}^y z_x
$$

$$
b^y(z_b) = W_{bz}^y z_b + b_0^y
$$

결과적으로 main LSTM의 gate 계산은 고정 weight만 사용하는 것이 아니라, hypernetwork가 만든 동적 scaling을 반영한다.

이것이 HyperLSTM의 핵심이다.

> LSTM이 hidden state만 업데이트하는 것이 아니라, 다음 step에서 사용할 계산 방식 자체를 조금씩 바꾼다.

## 7. Penn Treebank 결과

논문은 character-level Penn Treebank에서 HyperLSTM을 평가한다. 평가 지표는 bits-per-character, 즉 BPC다. 낮을수록 좋다.

| Model | Test BPC | Validation BPC | Param Count |
|---|---:|---:|---:|
| LSTM, 1000 units | 1.312 | 1.347 | 4.25M |
| LSTM, 1250 units | 1.306 | 1.340 | 6.57M |
| 2-Layer LSTM, 1000 units | 1.281 | 1.312 | 12.26M |
| Layer Norm LSTM, 1000 units | 1.267 | 1.300 | 4.26M |
| HyperLSTM, 1000 units | 1.265 | 1.296 | 4.91M |
| Layer Norm HyperLSTM, 1000 units | 1.250 | 1.281 | 4.92M |
| Layer Norm HyperLSTM, Large Embedding | 1.233 | 1.263 | 5.06M |
| 2-Layer Norm HyperLSTM | 1.219 | 1.245 | 14.41M |

HyperLSTM은 단순히 LSTM unit 수를 늘린 것보다 좋은 성능을 보인다. Layer Norm LSTM과 비슷하거나 더 좋고, Layer Norm과 결합하면 추가 성능 향상이 나온다.

여기서 중요한 해석은 이것이다.

```text
큰 LSTM을 쓰는 것보다,
작은 hypernetwork가 weight 조정 policy를 학습하는 것이 더 효율적일 수 있다.
```

## 8. enwik8 결과

논문은 더 큰 character modeling benchmark인 enwik8에서도 실험한다. enwik8은 100M character로 구성된 Hutter Prize Wikipedia dataset이다.

| Model | enwik8 BPC | Param Count |
|---|---:|---:|
| LSTM, 1800 units | 1.470 | 14.81M |
| LSTM, 2000 units | 1.461 | 18.06M |
| Layer Norm LSTM, 1800 units | 1.402 | 14.82M |
| HyperLSTM, 1800 units | 1.391 | 18.71M |
| Layer Norm HyperLSTM, 1800 units | 1.353 | 18.78M |
| Layer Norm HyperLSTM, 2048 units | 1.340 | 26.54M |

HyperLSTM은 Layer Norm LSTM보다 좋은 BPC를 얻고, Layer Norm과 결합하면 더 좋아진다.

![HyperNetworks loss curves]({{ '/assets/images/blog/hypernetworks-loss-curves.png' | relative_url }})

*왼쪽은 enwik8 validation BPC, 오른쪽은 handwriting generation validation log-loss다. HyperLSTM 계열은 기본 LSTM보다 빠르게 낮은 loss로 수렴한다.*

논문은 HyperLSTM이 단순히 최종 성능만 좋은 것이 아니라, training step 기준으로 더 빠르게 수렴한다고 설명한다.

## 9. 내부 동작: Layer Norm과 다른 방식

논문에서 특히 흥미로운 그림은 hidden state의 histogram이다.

![Hidden state histogram]({{ '/assets/images/blog/hypernetworks-hidden-state-histogram.png' | relative_url }})

*LSTM, Layer Norm LSTM, HyperLSTM, Layer Norm HyperLSTM의 hidden state 분포 비교. HyperLSTM은 Layer Norm과 비슷한 성능을 내지만 내부 분포는 꽤 다르다.*

Layer Norm LSTM은 saturation을 줄이는 방향으로 작동한다. activation을 정규화하기 때문에 값이 양끝에 과하게 몰리는 것을 완화한다.

그런데 HyperLSTM은 오히려 saturation이 강하게 나타난다. 그럼에도 성능은 Layer Norm LSTM과 비슷하거나 더 좋다.

이것은 중요한 관찰이다.

> HyperLSTM은 Layer Norm과 같은 일을 다른 방식으로 하는 것이 아니다. 아예 다른 동적 weight adjustment policy를 학습한다.

즉 HyperLSTM의 장점은 단순한 normalization 효과로 환원하기 어렵다.

## 10. Handwriting generation 결과

논문은 IAM Online Handwriting Database에서도 HyperLSTM을 평가한다. 이 task는 pen의 `(x, y)` 좌표와 pen-up/pen-down 신호를 예측하는 sequence generation 문제다.

![Handwriting table]({{ '/assets/images/blog/hypernetworks-handwriting-table.png' | relative_url }})

*IAM Online DB validation log-loss. HyperLSTM은 가장 낮은 log-loss를 기록한다.*

핵심 결과만 정리하면 다음과 같다.

| Model | Log-Loss | Param Count |
|---|---:|---:|
| LSTM, 900 units | -1055 | 3.36M |
| 2-Layer LSTM, 650 units | -1135 | 5.16M |
| Layer Norm LSTM, 900 units | -1096 | 3.37M |
| Layer Norm LSTM, 1000 units | -1106 | 4.14M |
| Layer Norm HyperLSTM, 900 units | -1067 | 3.95M |
| HyperLSTM, 900 units | -1162 | 3.94M |

여기서 재미있는 점은 Layer Norm HyperLSTM이 HyperLSTM보다 나쁘다는 것이다.

언어 모델링에서는 Layer Norm과 HyperLSTM의 조합이 좋았지만, handwriting generation에서는 그렇지 않았다. 논문은 이 결과를 두고, 통계 기반 normalization이 어떤 task에서는 hypernetwork가 학습한 weight adjustment policy에 방해가 될 수 있다고 해석한다.

이 부분이 꽤 중요하다.

```text
Layer Norm:
대부분의 sequence task에서 안정적인 기본기

HyperLSTM:
task에 따라 더 자유로운 dynamic policy를 학습

둘의 결합:
항상 좋은 것은 아님
```

## 11. Neural Machine Translation 결과

논문은 HyperLSTM을 대규모 neural machine translation에도 적용한다. GNMT 구조에서 LSTM cell을 HyperLSTM cell로 바꾸고, WMT'14 English-to-French task에서 평가한다.

| Model | Test BLEU | Log Perplexity |
|---|---:|---:|
| GNMT WPM-32K, LSTM | 38.95 | 1.027 |
| GNMT WPM-32K, ensemble of 8 LSTMs | 40.35 | - |
| GNMT WPM-32K, HyperLSTM | 40.03 | 0.993 |

단일 HyperLSTM 모델이 기존 단일 GNMT LSTM보다 높은 BLEU를 얻고, 8개 LSTM ensemble에 가까운 성능을 낸다.

이 결과는 HyperNetwork가 toy problem에만 적용되는 아이디어가 아니라, production-level sequence model에도 적용 가능하다는 점을 보여준다.

## 12. 무엇이 좋은가

이 논문의 가장 큰 장점은 관점 전환이다.

딥러닝에서 보통 weight는 학습된 후 고정된 함수의 일부로 쓰인다. 하지만 HyperNetwork는 weight 자체를 동적으로 생성할 수 있는 대상으로 본다.

```text
기존 neural network:
입력 x가 들어오면 고정된 weight W로 계산한다.

HyperNetwork:
입력과 상태를 보고 weight 또는 weight adjustment를 만든 뒤 계산한다.
```

특히 RNN/LSTM에 대해 이 관점은 강력하다.

RNN의 weight sharing은 시퀀스 모델링에서 중요한 inductive bias지만, 동시에 제약이다. HyperLSTM은 이 제약을 완전히 없애지 않고 부드럽게 완화한다.

또한 실험 범위가 넓다.

- MNIST
- CIFAR-10
- Penn Treebank
- enwik8
- IAM handwriting generation
- WMT'14 En-Fr translation

단순한 개념 증명에 머물지 않고 여러 종류의 task에서 작동함을 보여준다.

## 13. 한계점

첫 번째 한계는 복잡도다.

HyperLSTM은 기본 LSTM보다 구조가 훨씬 복잡하다. main LSTM 외에 hyper LSTM이 추가되고, embedding size, hyper cell size, scaling 방식, Layer Norm 적용 여부 같은 선택지가 늘어난다.

```text
기본 LSTM:
구현과 튜닝이 비교적 단순

HyperLSTM:
표현력은 늘지만 설계 공간도 커짐
```

두 번째 한계는 CNN 결과다.

Static HyperNetwork는 parameter 수를 줄이는 데는 효과적이지만, CIFAR-10 결과에서 정확도 손실이 있다. 따라서 CNN 쪽에서는 state-of-the-art 성능을 위한 방법이라기보다는 compression과 weight sharing 실험으로 보는 것이 자연스럽다.

세 번째 한계는 해석 가능성이다.

HyperLSTM이 어떤 scaling policy를 학습하는지 일부 histogram과 visualization을 보여주지만, 왜 특정 task에서 Layer Norm과 잘 결합되고 다른 task에서는 충돌하는지에 대한 이론적 설명은 충분하지 않다.

네 번째 한계는 계산 비용이다.

Full weight generation을 피하기 위해 scaling vector를 쓰긴 하지만, 기본 LSTM보다 추가 연산과 parameter가 필요하다. 따라서 실제 시스템에서는 성능 향상 대비 latency, memory, implementation complexity를 함께 봐야 한다.

## 14. 관련 아이디어와 연결

이 논문은 2016년 논문이지만, 지금 보면 여러 현대적인 아이디어와 연결된다.

```text
Dynamic parameter generation
Conditional computation
Meta-learning
Fast weights
Adapters
LoRA류의 low-rank parameter modulation
Mixture-of-Experts
```

물론 HyperNetwork와 LoRA나 MoE가 같은 것은 아니다. 하지만 공통된 큰 질문은 비슷하다.

> 모든 입력을 같은 고정 parameter로 처리해야 할까?

HyperNetwork는 이 질문에 대해 "parameter 자체를 입력과 문맥에 따라 생성하거나 조정하자"라고 답한다.

## 15. 내가 이해한 핵심

HyperNetworks를 처음 보면 "작은 network가 큰 network의 weight를 만든다"는 문장이 전부인 것처럼 보인다. 하지만 내가 이해한 핵심은 weight sharing 방식을 고정하지 않고 학습하게 만든다는 점이다.

더 중요한 메시지는 이것이다.

> weight sharing은 0 또는 1의 문제가 아니다. 완전히 공유할 수도 있고, 완전히 독립적으로 둘 수도 있고, HyperNetwork를 통해 그 사이를 학습할 수도 있다.

RNN은 모든 time step에서 weight를 공유한다. 이건 강한 가정이다. HyperLSTM은 이 가정을 완전히 버리지 않고, 작은 hypernetwork가 time step마다 weight scaling을 조정하게 만든다.

내가 보기에는 이 논문의 가장 좋은 부분은 성능표보다 이 관점이다.

```text
모델은 입력을 처리하는 함수다.
그런데 그 함수 자체도 입력과 상태에 따라 바뀔 수 있다.
```

이 생각은 지금 봐도 꽤 신선하다. 특히 LLM 시대에는 adapter, routing, conditional computation, test-time adaptation처럼 "고정된 모델을 어떻게 상황에 맞게 바꿀 것인가"가 계속 중요해지고 있다.

그런 의미에서 HyperNetworks는 오래된 논문이지만, 여전히 읽을 가치가 있는 논문이다.
