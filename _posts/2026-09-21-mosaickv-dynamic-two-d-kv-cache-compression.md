---
title: "MosaicKV 정리: 긴 문맥 LLM을 위한 동적 2차원 KV Cache 압축"
date: 2026-09-21 00:00:00 +0900
last_modified_at: 2026-09-21 00:00:00 +0900
summary: "MosaicKV가 토큰과 채널을 함께, 그러나 고정된 규칙이 아닌 세그먼트별·벡터별 규칙으로 압축해 long-context LLM의 KV cache 병목을 줄이는 방식을 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - Serving
  - KV Cache
  - Long Context
  - Attention
  - Compression
  - Paper
source: "arXiv:2607.00760"
---

# MosaicKV: 긴 문맥 LLM의 KV Cache를 동적으로 2차원 압축하기

긴 문맥 LLM 서빙에서 가장 먼저 부딪히는 문제는 모델 파라미터가 아니라 **KV cache**다. 입력 길이와 동시 요청 수에 따라 KV cache는 선형으로 커지고, decode 단계에서는 매 토큰마다 긴 cache를 읽어야 한다. 그래서 메모리 용량뿐 아니라 memory bandwidth도 병목이 된다.

[MosaicKV: Serving Long-Context LLM with Dynamic Two-D KV Cache Compression](https://arxiv.org/abs/2607.00760)는 이 문제를 토큰 축과 채널 축 양쪽에서 줄인다. 다만 단순히 둘을 겹쳐 압축하지 않는다. 각 KV vector에서 중요한 원소를 따로 고르고, KV cache를 여러 segment로 나누어 segment마다 압축 규칙을 다르게 정한다.

> MosaicKV의 핵심은 “모든 token과 모든 channel에 같은 압축률을 적용하지 말자”이다.

논문은 H800 환경에서 Full KV cache 대비 최대 3배의 메모리 절감, 4.8배 낮은 decode latency, 7.3배 높은 decode throughput을 보고한다. LongBench와 RULER의 평균 정확도 손실은 1.76%다. 이 글은 왜 2차원 압축이 어려운지와 MosaicKV가 이를 어떻게 풀었는지에 초점을 맞춘다.

## 1. 왜 KV cache가 long context의 병목이 되는가

autoregressive LLM은 다음 토큰을 생성할 때 이전 토큰들의 `Key`, `Value`를 다시 계산하지 않도록 KV cache에 저장해 재사용한다. 따라서 cache 크기는 대략 다음에 비례한다.

$$
KV\ Cache\ Memory \propto B \times S \times L \times H_{kv} \times D
$$

- $B$: batch size
- $S$: sequence length
- $L$: layer 수
- $H_{kv}$: KV head 수
- $D$: head dimension

여기서 중요한 것은 $S$와 $B$가 함께 커진다는 점이다. 논문은 LLaMA-3.1-8B에서 128K token 요청 8개를 동시에 서빙하면 KV cache만 128GB가 필요하며, 1M token에서는 약 1TB까지 커진다고 든다. 모델 weight보다 KV cache가 훨씬 큰 비용이 될 수 있는 이유다.

또한 decode에서는 새 query 하나가 과거 전체 K/V를 읽는다. 논문 프로파일링에서는 1M context에서 attention이 token 간 지연 시간(Time-Between-Tokens)의 97.9%를 차지했다. cache를 덜 저장하는 것과 덜 읽는 것은 곧바로 서빙 성능과 연결된다.

## 2. 기존 압축은 한 축만 잘 줄인다

KV cache를 줄이는 방식은 크게 두 종류다.

| 방식 | 줄이는 대상 | 대표적인 동작 | 약점 |
| --- | --- | --- | --- |
| Sequence compression | token / token block | 현재 query와 관련 큰 token block만 top-k로 선택 | 선택 후보를 위해 전체 cache를 계속 보관해야 할 수 있음 |
| Channel compression | 각 K/V vector의 dimension | 중요 channel 또는 원소만 남김 | 모든 vector에 같은 mask를 쓰면 중요한 원소를 버릴 수 있음 |

Quest 같은 sequence compression은 query와 block key의 score를 계산한 뒤 중요한 block만 attention에 넣는다. ThinK류의 channel compression은 미리 중요한 channel을 정해 K/V vector의 폭을 줄인다.

둘을 합치면 곱셈 효과를 기대할 수 있다. 예를 들어 token의 10%만 고르고 각 vector의 25% 원소만 남기면 attention이 다뤄야 할 K/V 양이 크게 줄어든다. 하지만 논문이 만든 단순 결합(strawman)은 channel compression 30%에서 정확도가 24.5% 떨어지고, 70%에서는 82.8%까지 떨어졌다.

문제는 **중요도가 균일하지 않다**는 것이다. SVD rotation 뒤에도 모든 vector의 중요한 원소가 같은 channel에 모이지 않는다. 논문에서 vector별 top 25% 중요 원소 중 전역 top 25% channel 안에 드는 비율은 62.28%에 그쳤다. 하나의 전역 mask는 나머지 중요한 원소를 버린다.

## 3. MosaicKV의 관점: 같은 cache라도 압축하기 쉬운 곳이 다르다

MosaicKV는 KV cache를 시간 순서의 여러 segment로 나눈다. 그리고 각 segment에 대해 K와 V를 따로 회전하고, channel 압축률·원소 묶음 크기·token block 크기를 고른다.

![MosaicKV의 동적 2차원 압축 과정](/assets/images/blog/mosaickv-dynamic-two-d-compression.png)

*각 segment는 별도의 SVD rotation과 channel/sequence 압축 전략을 사용한다. 출처: MosaicKV 논문 Figure 5.*

이 선택이 필요한 이유는 segment마다 분포가 다르기 때문이다.

- 큰 outlier가 일부 원소에 뚜렷이 모인 segment는 높은 channel 압축률을 써도 된다.
- 원소 크기가 비교적 고른 segment는 더 많은 원소를 남겨야 한다.
- token별로도 중요한 channel이 달라서, 전역 channel mask 대신 vector별 top-$r$ 원소를 선택해야 한다.

즉 MosaicKV는 하나의 고정 sparsity pattern이 아니라, cache가 가진 분포에 맞춰 달라지는 작은 압축 정책들의 모음이다. 이름의 Mosaic은 이처럼 서로 다른 압축 조각이 하나의 KV cache를 이룬다는 의미로 이해하면 쉽다.

## 4. 두 단계로 하는 2차원 압축

### Stage 1: vector마다 중요한 원소만 남긴다

먼저 segment의 K, V 행렬에 각각 SVD 기반 rotation을 적용한다. rotation은 큰 정보를 몇몇 축에 더 잘 모이게 하는 전처리다. 그 다음 **각 KV vector별로** 절댓값이 큰 top-$r$ 원소를 선택한다.

```text
고정 channel mask:
    모든 vector에서 같은 32개 channel만 보존

MosaicKV:
    vector A에서는 {2, 9, 31, ...}
    vector B에서는 {1, 8, 17, ...}
    처럼 각 vector의 top-r 원소를 보존
```

이 단계는 K, V뿐 아니라 나중에 token block을 고르는 데 쓰는 per-block key에도 적용된다. 결과적으로 score 계산과 최종 attention 모두 압축된 representation 위에서 수행할 수 있다.

### Stage 2: 현재 query에 필요한 token block만 고른다

각 segment의 token을 8개나 16개 같은 작은 block으로 나누고, block 안 key의 평균으로 per-block key를 만든다. 현재 query와 per-block key의 내적을 계산해 top-$k$ block을 고른 뒤, 그 block의 KV vector만 실제 attention에 사용한다.

```text
현재 query
  -> 압축된 per-block key와 score 계산
  -> top-k token block 선택
  -> 선택된 block의 압축된 K/V로 attention
```

첫 단계가 vector의 폭을 줄이고, 두 번째 단계가 읽을 token 수를 줄인다. 그래서 “2차원” 압축이다.

## 5. 압축률만 높여서는 빨라지지 않는다

원소별로 다른 mask를 쓰면 sparse matrix가 된다. 하지만 GPU는 무작위 sparse access보다 연속된 dense access에 훨씬 잘 맞는다. 메모리를 75% 줄였더라도 위치를 찾아가며 읽느라 cache miss가 늘면 오히려 느려질 수 있다.

MosaicKV는 이 문제를 **packed KV format**으로 푼다. 남긴 원소를 vector마다 연속된 메모리에 저장하고, 원래 위치는 bitmap으로 표시한다. attention kernel은 K/V를 원래 dense tensor로 복원하지 않은 채 bitmap을 따라 packed vector와 query의 해당 원소를 곱한다.

```text
원래 vector: [k0, k1, k2, k3, k4, k5]
bitmap:      [ 0,  1,  1,  0,  1,  0]
packed K:             [k1, k2, k4]

attention: bitmap을 보며 query의 q1, q2, q4만 꺼내 packed K와 곱함
```

원소마다 index를 저장하는 COO/CSR 형식 대신 bitmap을 쓰는 것도 중요하다. 이 bitmap 자체의 비용을 줄이기 위해 인접한 원소를 1·2·4개 그룹으로 함께 선택하는 가변 granularity도 사용한다.

## 6. decode 경로를 막지 않는 cache 관리

새로운 output token은 decode 중 하나씩 생긴다. 매 token마다 SVD를 다시 계산하고 segment 정책을 갱신하면 압축 관리가 attention 이득을 잡아먹을 수 있다.

MosaicKV는 이를 GPU와 CPU의 역할 분리로 숨긴다.

- GPU update buffer에는 새 K/V를 임시 압축해 두고, decode는 즉시 진행한다.
- CPU buffer는 segment 분할, SVD rotation, 최종 압축 전략 선택처럼 상대적으로 무거운 작업을 비동기로 수행한다.
- 최종 압축이 준비되면 임시 segment를 교체한다.

논문은 decode attention이 memory bandwidth에 치우쳐 CUDA core와 CPU가 남는다는 관찰을 출발점으로 삼는다. 그래서 packed sparse attention은 남는 CUDA core를, 압축 관리 작업은 CPU를 활용한다. 알고리즘의 sparsity와 실제 시스템의 자원 사용을 함께 설계한 부분이다.

## 7. 실험에서 무엇이 달라졌나

논문은 H800 80GB에서 LLaMA-3.1-8B, Qwen3-4B, Ministral-3-8B를 평가했다. Full KV cache와 sequence-only Quest가 기준선이다.

| 지표 | MosaicKV 결과 | 비교 기준 |
| --- | ---: | --- |
| KV cache memory | 최대 3배 감소 | Full KV cache |
| attention latency | 최대 16배 빠름 | Full |
| decode latency (1M context) | 최대 4.8배 낮음 | vLLM Full |
| decode throughput (1M context) | 최대 7.3배 높음 | vLLM Full |
| LongBench + RULER 정확도 손실 | 평균 1.76% | Full |

1M context에서 MosaicKV는 vLLM/Quest가 batch size 1에서도 전체 KV cache를 수용하기 어려운 H800 환경에서 약 1.44M token까지 확장했다고 보고한다. 25% channel retention 설정에서는 Full보다 KV cache가 3배, Quest보다 3.18배 작았다.

정확도도 단순히 평균 하나만 볼 일은 아니다. RULER와 LongBench의 일부 세부 task에서는 하락이 있으며, 모델별 양상도 다르다. 예를 들어 RULER 평균은 LLaMA-3.1-8B에서 Full 91.56, MosaicKV 89.85이고, Ministral-3-8B에서는 Full 90.05, MosaicKV 86.94다. 따라서 “1.76% 평균 손실”을 모든 workload에서 같은 품질이라는 뜻으로 읽으면 안 된다.

## 8. 이 논문이 실무에 주는 의미

MosaicKV는 KV cache를 단순한 저장 공간이 아니라 **정확도·메모리·bandwidth의 균형을 계속 조절해야 하는 데이터 구조**로 본다. 특히 다음 상황에서 의미가 크다.

- 코드베이스, 다수 문서, 대화 이력을 길게 넣는 서비스
- context 길이와 동시성이 함께 큰 워크로드
- GPU 메모리 부족 때문에 batch size를 올리지 못하는 서빙 환경
- token selection만으로는 cache 저장량을 충분히 줄이기 어려운 경우

다만 즉시 범용 runtime에 적용할 수 있는 기법으로 보기는 이르다. 논문의 구현은 PyTorch 약 5,300줄과 CUDA 약 2,800줄의 prototype이며, 성능 수치는 H800과 특정 모델·설정에서 얻은 결과다. packed sparse attention kernel, bitmap overhead, segment 정책의 threshold, prefill overhead가 실제 모델과 GPU에서 어떻게 바뀌는지 검증이 필요하다.

또한 MosaicKV는 PagedAttention, RadixAttention, FlashAttention과 양립 가능하다고 설명한다. PagedAttention과 RadixAttention이 cache의 할당·공유를 다룬다면, MosaicKV는 cache **내용 자체를 더 작게 표현하고 읽는 방식**을 다룬다. 서로 경쟁하기보다 조합 가능한 최적화에 가깝다.

## 마무리

MosaicKV의 기여는 “토큰도 줄이고 channel도 줄이자”가 아니다. 그것만으로는 정확도가 크게 망가진다는 사실에서 시작한다.

```text
정적 2차원 압축
    = 모두에게 같은 token/channel 압축 규칙
    -> 중요한 정보까지 함께 버리기 쉬움

MosaicKV의 동적 2차원 압축
    = segment마다 다른 정책 + vector마다 다른 중요 원소
    -> 높은 압축률과 낮은 정확도 손실을 함께 노림
```

그리고 sparse cache를 만든 뒤 packed format, 전용 attention, 비동기 buffer까지 연결해 “압축했지만 느린” 상태를 피한다. context window가 수십만에서 수백만 token으로 커질수록, 이런 데이터 분포 인식형 KV cache 최적화는 더 중요한 서빙 계층이 될 가능성이 크다.

## 참고 자료

- [MosaicKV: Serving Long-Context LLM with Dynamic Two-D KV Cache Compression](https://arxiv.org/abs/2607.00760)
- [MosaicKV PDF](https://arxiv.org/pdf/2607.00760)
- [Long Context에서 KV Cache가 병목이 되는 이유](/engineering/2026/06/21/large-kv-cache-long-context.html)
- [vLLM과 PagedAttention 정리](/engineering/2026/05/29/vllm.html)
