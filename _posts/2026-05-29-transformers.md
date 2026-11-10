---
title: "Transformer 정리: Attention만으로 시퀀스를 다루는 방법"
date: 2026-05-29 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - study
tags:
  - AI
  - LLM
  - Transformer
  - Attention
source: "Notion - Transformers"
---

Transformer는 Google의 논문 **Attention Is All You Need**에서 제안된 모델이다. 기존의 seq2seq처럼 encoder-decoder 구조를 사용하지만, RNN 계열의 recurrent 구조 없이 attention만으로 시퀀스를 처리한다.

핵심은 문장을 순서대로 하나씩 읽는 대신, 문장 안의 토큰들이 서로 어떤 관계를 갖는지 직접 계산한다는 점이다.

## 기존 Sequence Modeling의 한계

Transformer 이전에는 RNN, LSTM, seq2seq 같은 구조가 많이 사용됐다.

RNN은 이전 hidden state를 다음 step으로 넘기면서 시퀀스를 처리한다. 이 구조는 순서를 자연스럽게 반영할 수 있지만, 긴 문장을 처리할 때 앞쪽 정보가 희미해지고 병렬화가 어렵다. 각 토큰을 순차적으로 계산해야 하므로 메모리와 연산 비용에서도 한계가 있다.

seq2seq는 encoder가 입력 문장을 하나의 context vector로 압축하고, decoder가 그 벡터를 바탕으로 출력 문장을 생성한다. 하지만 고정 크기의 context vector에 긴 문장의 정보를 모두 담아야 하므로 병목이 생길 수 있다.

Transformer는 이 문제를 attention으로 풀었다. 토큰끼리의 관계를 직접 계산해 필요한 정보를 가져오도록 만든 것이다.

![Transformer 원문 Figure: scaled dot-product attention과 multi-head attention]({{ '/assets/images/blog/transformer-paper-figure.png' | relative_url }})

*원 논문 Figure 2는 scaled dot-product attention과 multi-head attention의 구조를 함께 보여준다.*

## Encoder와 Decoder 구조

Transformer는 크게 encoder와 decoder로 나뉜다.

![transformer architecture]({{ '/assets/images/blog/transformer-architecture.png' | relative_url }})
![transformer block]({{ '/assets/images/blog/transformer-block.png' | relative_url }})

Encoder는 입력 문장을 이해해서 의미 벡터로 바꾼다. 각 encoder layer는 self-attention과 feed-forward network로 구성되고, 각 sub-layer에는 residual connection과 layer normalization이 붙는다.

Decoder는 encoder가 만든 정보를 바탕으로 단어를 하나씩 생성한다. decoder에는 masked self-attention, encoder-decoder attention, feed-forward network가 들어간다.

논문에서는 encoder와 decoder를 각각 6층 쌓고, 모델 차원은 512로 사용했다.

## Attention의 직관

Attention은 Query, Key, Value로 설명할 수 있다.

예를 들어 `I love her`를 `나는 그녀를 사랑한다`로 번역한다고 하자. 특정 출력 토큰을 만들 때, 모델은 입력 문장의 어떤 단어를 더 참고해야 하는지 계산해야 한다.

- Query: 지금 알고 싶은 대상
- Key: 비교 대상
- Value: 실제로 가져올 정보

Query와 Key의 유사도를 계산하면 어떤 토큰을 더 봐야 하는지 알 수 있다. 그 유사도에 따라 Value를 가중합하면 attention 출력이 된다.

즉 attention은 "지금 토큰을 이해하거나 생성하기 위해 문장 안의 어떤 정보를 얼마나 참고할 것인가"를 계산하는 방식이다.

## Scaled Dot-Product Attention

Scaled dot-product attention은 다음 흐름으로 동작한다.

![scaled dot product attention]({{ '/assets/images/blog/scaled-dot-product-attention.png' | relative_url }})
![transformer attention math]({{ '/assets/images/blog/transformer-attention-math.png' | relative_url }})

1. 입력 embedding에서 Q, K, V를 만든다.
2. Q와 K를 내적해 attention score를 구한다.
3. score를 $\sqrt{d_k}$로 나눈다.
4. softmax로 확률 분포를 만든다.
5. 이 확률을 V에 곱해 최종 attention 값을 만든다.

$\sqrt{d_k}$로 나누는 이유는 내적 값이 너무 커지는 것을 막기 위해서다. 값이 커지면 softmax가 한쪽으로 치우치고 gradient가 작아져 학습이 불안정해질 수 있다.

## Multi-Head Attention

Multi-head attention은 attention을 한 번만 수행하지 않고 여러 head로 나누어 병렬로 수행한다.

![multi head attention]({{ '/assets/images/blog/multi-head-attention.png' | relative_url }})
![attention vs mha]({{ '/assets/images/blog/attention-vs-mha.png' | relative_url }})

모델 차원이 `d`이고 head 수가 `h`라면, 각 head는 `d / h` 차원에서 attention을 계산한다. 이후 각 head의 결과를 concat하고 linear layer를 통과시킨다.

이렇게 나누는 이유는 서로 다른 head가 서로 다른 관계를 학습할 수 있기 때문이다. 어떤 head는 주어와 동사의 관계를 보고, 다른 head는 목적어나 위치 정보를 볼 수 있다. 여러 의미 공간에서 관계를 본 뒤 다시 합치는 구조다.

## Attention의 종류

Transformer 안에서는 attention이 여러 방식으로 쓰인다.

![masked self attention]({{ '/assets/images/blog/masked-self-attention.png' | relative_url }})

Encoder self-attention은 encoder 입력 안에서 토큰끼리 서로를 참고한다. `I love her`라는 문장이 있다면 각 토큰이 문장 안의 다른 모든 토큰을 볼 수 있다.

Masked decoder self-attention은 decoder에서 사용된다. 생성 모델은 다음 단어를 예측할 때 미래 토큰을 보면 안 된다. 그래서 현재 위치 이후의 토큰을 mask 처리한다.

Encoder-decoder attention은 decoder가 encoder 출력을 참고하는 부분이다. decoder의 query가 encoder의 key, value를 바라보면서 입력 문장과 출력 문장의 관계를 학습한다.

## Positional Encoding
![positional encoding]({{ '/assets/images/blog/positional-encoding.png' | relative_url }})

Transformer는 RNN처럼 순서대로 토큰을 처리하지 않는다. attention만 사용하면 토큰의 위치 정보가 사라진다. `나는 그녀를 사랑한다`와 `그녀는 나를 사랑한다`는 단어는 비슷하지만 순서에 따라 의미가 달라진다.

그래서 Transformer는 embedding에 positional encoding을 더한다. 논문에서는 sin, cos 함수를 사용해 위치 정보를 만든다.

이 방식은 학습 중 보지 못한 더 긴 시퀀스에도 어느 정도 일반화할 수 있다는 장점이 있다.

## 내가 이해한 핵심

Transformer는 recurrent 구조의 병렬화 한계와 context vector 병목을 attention으로 해결한 모델이다.

중요한 포인트는 다음과 같다.

- attention은 토큰 간 관계를 직접 계산한다.
- multi-head attention은 여러 관점에서 관계를 학습한다.
- decoder는 미래 토큰을 보지 않도록 masked attention을 사용한다.
- positional encoding으로 순서 정보를 보완한다.

LLM을 이해하려면 Transformer의 attention 구조를 먼저 잡는 것이 중요하다. 이후 GPT 계열 모델, encoder-only 모델, encoder-decoder 모델의 차이를 볼 때도 이 구조가 기준점이 된다.
