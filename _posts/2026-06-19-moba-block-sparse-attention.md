---
title: "MoBA: Long-Context LLM을 위한 Block Sparse Attention 이해하기"
date: 2026-06-19 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - Transformer
  - Attention
  - Serving
source: "arXiv:2502.13189, MoonshotAI/MoBA"
---

# MoBA: Long-Context LLM을 위한 Block Sparse Attention 이해하기

이 글은 논문 [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)와 [MoonshotAI/MoBA 공식 GitHub](https://github.com/MoonshotAI/MoBA)를 바탕으로 정리한 글이다.

LLM에서 context length를 늘리는 일은 매력적이다. 긴 문서, 여러 파일로 된 코드베이스, 긴 대화 기록, 대규모 RAG 결과를 한 번에 넣을 수 있기 때문이다.

하지만 Transformer의 standard full attention은 sequence length가 길어질수록 비용이 빠르게 커진다. 이 글에서는 MoBA(Mixture of Block Attention)가 이 문제를 어떻게 풀려고 하는지, Sparse Attention과 Linear Attention과는 무엇이 다른지, 그리고 실제 LLM serving 관점에서 어떤 trade-off가 있는지 정리한다.

논문의 핵심은 간단하다. 모든 token을 모든 token에 attention하지 말고, sequence를 block으로 나눈 뒤 query마다 관련 있는 block만 골라 attention하자는 것이다. MoBA는 이를 위해 block-level routing과 top-k gating을 사용한다.

![MoBA 논문 Figure 1a: Full Attention, Sparse Attention, MoBA의 비교]({{ '/assets/images/blog/moba-fig1a.webp' | relative_url }})

*MoBA는 모든 token을 보는 full attention과 달리, query별로 관련 block을 선택해 attention을 수행한다.*

## 1. 왜 Long-Context Attention이 문제가 되는가

Transformer의 standard self-attention은 모든 query가 모든 key/value를 본다. causal LM에서는 미래 token을 masking하긴 하지만, 현재 token 기준으로는 이전 token 전체를 볼 수 있는 구조다.

```text
Q1 -> K1
Q2 -> K1 K2
Q3 -> K1 K2 K3
Q4 -> K1 K2 K3 K4
...
Qn -> K1 K2 K3 ... Kn
```

scaled dot-product attention은 보통 다음처럼 쓴다.

$$
Attention(Q, K, V)
= softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

여기서 `QK^T`는 sequence length가 `n`이면 대략 `n x n` 크기의 attention score matrix를 만든다. 그래서 full attention의 계산량은 `O(n^2)`로 커진다.

context가 짧을 때는 이 구조가 강력하다. 모델이 필요한 정보를 어디서든 가져올 수 있기 때문이다. 하지만 context가 길어지면 병목이 커진다.

- prefill latency가 증가한다.
- attention score 계산과 memory access 비용이 커진다.
- KV cache가 길어져 VRAM 사용량이 늘어난다.
- 동시에 처리 가능한 request 수가 줄어 serving cost가 올라간다.

특히 LLM serving에서는 prefill과 decode를 나눠서 생각해야 한다.

```text
긴 prompt 입력
   |
   v
prefill: prompt 전체를 한 번에 처리하고 KV cache 생성
   |
   v
decode: KV cache를 재사용하며 token을 하나씩 생성
```

long-context 요청에서는 prefill 단계가 특히 무겁다. prompt token 전체가 attention 계산에 참여하기 때문이다. decode 단계에서도 KV cache가 길어지면 매 step에서 참조해야 할 cache가 많아지고, memory bandwidth와 cache management가 중요해진다.

## 2. Sparse Attention과 Linear Attention의 차이

long-context attention을 줄이려는 방법은 크게 두 방향으로 볼 수 있다.

하나는 **보는 대상을 줄이는 것**이고, 다른 하나는 **attention 계산식 자체를 바꾸는 것**이다.

### Sparse Attention: 일부 token이나 block만 보기

Sparse attention은 모든 token을 다 보지 않는다. 미리 정한 패턴이나 동적으로 선택한 subset만 본다.

대표적인 예시는 window attention이다.

```text
Window attention

... K7 K8 K9 [Q10] K11 K12 K13 ...
        ^^^^^ 현재 query 주변 window만 attention
```

causal LM에서는 보통 미래 token을 볼 수 없으므로, 현재 token 기준으로 과거의 일정 window만 보는 식이 된다.

또 다른 예시는 block sparse attention이다.

```text
Sequence blocks

[B1] [B2] [B3] [B4] [B5]
          ^
          query가 있는 위치

선택된 block: [B1], [B3]
무시된 block: [B2], [B4], [B5]
```

Sparse attention의 장점은 직관적이다.

- full attention보다 계산량과 memory 사용량을 줄일 수 있다.
- local context가 중요한 작업에서는 window attention만으로도 효율적일 수 있다.
- block 단위로 sparse pattern을 만들면 GPU kernel 최적화와 결합할 여지가 있다.

하지만 단점도 분명하다.

- 중요한 정보가 sparse pattern 밖에 있으면 놓칠 수 있다.
- window나 block pattern이 task-specific bias를 만들 수 있다.
- pattern이 고정되어 있으면 query마다 필요한 정보가 달라도 유연하게 대응하기 어렵다.

### Linear Attention: attention 계산식을 바꾸기

Linear attention은 sparse attention과 출발점이 다르다. sparse attention이 "무엇을 볼 것인가"를 줄이는 방식이라면, linear attention은 "attention을 어떻게 계산할 것인가"를 바꾸는 방식이다.

```text
Sparse attention
Q -> 일부 K/V 선택 -> softmax attention

Linear attention
Q, K, V -> attention 계산식 변형 -> O(n^2) 완화
```

즉, sparse attention은 기존 softmax attention의 범위를 줄이는 쪽에 가깝고, linear attention은 softmax attention을 근사하거나 다른 형태로 바꿔 quadratic cost를 줄이려는 쪽에 가깝다.

이 차이는 중요하다. Linear attention은 이론적으로 긴 sequence를 더 효율적으로 처리할 수 있지만, 기존 softmax attention과 동작 특성이 달라질 수 있다. 논문도 linear approximation 계열이 기존 Transformer와 달라질 수 있고, 복잡한 reasoning 성능이 충분히 검증되지 않았다는 취지로 한계를 언급한다.

짧게 말하면 다음과 같다.

```text
Sparse Attention = 어디를 볼지 줄인다
Linear Attention = attention 계산 방식을 바꾼다
MoBA             = block 단위로 어디를 볼지 query마다 고른다
```

## 3. MoBA의 핵심 아이디어

MoBA의 핵심은 attention에 MoE-style routing을 넣는 것이다. Expert 대신 block을 고른다고 생각하면 이해하기 쉽다.

```text
긴 context
   |
   v
[Block 1] [Block 2] [Block 3] [Block 4] ...
   |          |          |          |
   v          v          v          v
 block key  block key  block key  block key
   \          |          |          /
    \         |          |         /
          query와 score 계산
                  |
                  v
             top-k block 선택
                  |
                  v
       선택된 block 내부 token K/V attention
```

절차는 다음과 같다.

1. sequence를 고정 길이 block으로 나눈다.
2. 각 block 안의 key vector들을 평균내어 block 대표 key vector를 만든다.
3. query vector와 각 block 대표 key vector의 dot product로 routing score를 계산한다.
4. score가 높은 top-k block을 선택한다.
5. 선택된 block 내부의 실제 token-level K/V에 대해 attention을 수행한다.

여기서 중요한 점은 block 대표 key가 routing에만 쓰인다는 것이다. 최종 attention은 block 평균 vector에 하는 것이 아니라, 선택된 block 안에 있는 실제 token K/V에 대해 수행한다.

```text
Routing 단계:
q · mean(K_block)

Attention 단계:
q attends to K_token, V_token inside selected blocks
```

이 구조는 full attention과 sparse attention 사이의 중간 지점에 있다. 모든 token을 다 보지는 않지만, query마다 다른 block을 선택할 수 있기 때문에 고정 window보다 유연하다.

![MoBA 논문 Figure 1b: query와 block routing 구조]({{ '/assets/images/blog/moba-fig1b.webp' | relative_url }})

*MoBA는 query와 block 대표 key 사이의 score를 이용해 관련 block을 고른 뒤, 선택된 block 내부의 실제 K/V에 attention한다.*

## 4. Top-k Gating은 어떻게 동작하는가

MoBA의 gating은 별도 parameter가 있는 router network라기보다, query와 block 대표 key의 similarity를 이용하는 parameter-less gating에 가깝다. 공식 README도 이 점을 `Parameter-less Gating Mechanism`이라고 설명한다.

예를 들어 query `q`가 있고, context가 5개 block으로 나뉘어 있다고 하자.

```text
q
|
+--> score(q, B1)
+--> score(q, B2)
+--> score(q, B3)
+--> score(q, B4)
+--> score(q, B5)

top-k = 2
|
v
B2, B4 선택
```

선택 이후에는 `B2`, `B4` 내부의 실제 token key/value에 대해 attention을 계산한다.

causal LM에서는 미래 정보를 보면 안 된다. 그래서 MoBA는 causal constraint도 함께 처리한다.

```text
[B1] [B2] [B3] [B4] [B5]
           ^
       current block

가능한 block: B1, B2, B3
불가능한 block: B4, B5
항상 포함: B3(current block)
```

current block을 항상 포함하는 이유는 현재 query 주변의 local context를 안정적으로 보장하기 위해서다. 다만 current block 내부에서도 미래 token은 볼 수 없으므로 causal mask가 필요하다.

```text
current block 내부

t1  t2  t3  t4
        ^
      query

볼 수 있음: t1, t2, t3
볼 수 없음: t4
```

이렇게 보면 MoBA의 routing은 꽤 실용적인 타협이다. 먼 과거의 모든 block을 다 보지는 않지만, query가 필요하다고 판단한 block은 고를 수 있다. 동시에 current block은 항상 포함해 local continuity를 유지한다.

## 5. MoBA는 왜 추가 학습이 필요한가

MoBA를 볼 때 가장 조심해야 할 부분이 있다. MoBA는 pretrained model에 attention module만 바꿔 끼우고 바로 inference하는 trick이 아니다.

공식 README는 MoBA에 continue training이 필요하다고 설명한다. 즉, 기존 모델이 full attention으로 학습되어 있다면, attention architecture를 MoBA로 바꾼 뒤 모델이 sparse block routing 구조에 적응하도록 추가 학습이 필요하다.

왜 그럴까?

Full attention으로 학습된 모델은 모든 이전 token을 볼 수 있다는 전제 위에서 internal representation을 만들었다. 그런데 MoBA로 바꾸면 query가 일부 block만 보게 된다. 모델 입장에서는 정보 접근 방식 자체가 달라지는 셈이다.

```text
기존 full attention 모델
  "필요하면 모든 과거 token을 볼 수 있음"

MoBA 모델
  "routing으로 선택된 block과 current block을 중심으로 봄"
```

따라서 모델은 다음을 새로 익혀야 한다.

- 어떤 query가 어떤 block을 선택해야 하는지
- block 평균 key 기반 routing이 실제 token-level attention과 어떻게 연결되는지
- sparse하게 본 정보로도 다음 token 예측을 안정적으로 수행하는 방법

이 점에서 MoBA는 inference 단계의 단순 최적화라기보다 attention architecture 자체의 변화다.

## 6. 기술적 한계와 Trade-off

MoBA는 long-context attention을 줄이는 흥미로운 접근이지만, 공짜는 아니다. 설계 선택마다 trade-off가 생긴다.

### top-k 선택

`top-k`가 작으면 attention해야 할 block 수가 줄어든다. 속도와 memory 측면에서는 유리하다.

하지만 중요한 block이 top-k 안에 들어오지 못하면 모델이 필요한 정보를 놓칠 수 있다.

```text
top-k 작음
  빠름
  중요한 block 누락 위험 증가

top-k 큼
  안정적
  full attention에 가까워져 이점 감소
```

### block size 선택

block size도 중요하다.

block size가 작으면 routing이 더 정밀해진다. 특정 query가 필요한 작은 구간만 선택할 가능성이 커진다. 대신 block 수가 많아지고, block 대표 key 계산과 top-k routing overhead가 늘어난다.

block size가 크면 GPU 효율과 kernel 구현 측면에서 유리할 수 있다. 하지만 block 내부의 key를 평균내는 과정에서 중요한 token 정보가 희석될 수 있다.

```text
작은 block
  정밀한 routing
  routing overhead 증가

큰 block
  GPU 효율에 유리할 수 있음
  평균 pooling에서 token 정보 희석 가능
```

### FlashAttention 대비 구현 난이도

FlashAttention은 full attention의 결과를 유지하면서 GPU memory IO를 줄이는 최적화다. 이미 많은 LLM inference stack에서 중요한 기반 기술로 쓰인다.

반면 MoBA는 어떤 block을 볼지 선택하는 routing이 들어간다. 따라서 단순히 attention kernel만 빠르게 만드는 문제를 넘어선다.

- block metadata 관리가 필요하다.
- top-k routing 결과에 따라 attention 대상이 달라진다.
- causal mask와 current block 포함 규칙을 함께 처리해야 한다.
- 기존 serving engine의 KV cache layout과 맞물려야 한다.

공식 GitHub는 CUDA kernel 구현과 flash-attn 의존성을 제공하지만, 실제 production serving stack에 넣는 일은 별도의 engineering 작업이 필요할 수 있다. 공개 자료 기준으로는 vLLM이나 SGLang에 공식 통합되었다고 단정하기 어렵다.

### 짧은 context에서는 이득이 작을 수 있음

MoBA의 장점은 long-context에서 커진다. 반대로 context가 짧으면 full attention 자체가 이미 충분히 빠를 수 있다. 이 경우 routing overhead가 상대적으로 커져 MoBA의 이점이 작을 수 있다.

이 부분은 workload, sequence length, batch size, GPU, kernel 구현에 따라 달라진다. 공개 자료만으로 모든 serving 환경에서의 우위를 단정하기는 어렵다.

## 7. LLM Serving 관점에서의 의미

LLM serving에서 자주 나오는 키워드는 `vLLM`, `SGLang`, `KV cache`, `prefix cache`, `prefill latency`다.

MoBA는 이 키워드들과 직접 연결해서 생각해볼 만하다.

### prefill latency

긴 prompt를 처리할 때 prefill은 무겁다. full attention은 prompt 내부 token들이 서로 넓게 attention해야 하므로 context가 길수록 비용이 커진다.

MoBA-style routing이 잘 동작한다면, 모든 block을 dense하게 보는 대신 query별로 관련 block만 보게 되어 prefill 비용을 줄일 가능성이 있다. 다만 실제 latency 개선 폭은 구현과 workload에 의존하므로, 공개 자료만으로 일반화해서 말하기는 어렵다.

### KV cache

vLLM의 PagedAttention처럼 KV cache를 block 단위로 관리하는 접근은 serving에서 이미 중요하다. MoBA도 context를 block 단위로 나누고 routing한다는 점에서, KV cache 관리와 연결해볼 여지가 있다.

예를 들어 연구 아이디어로는 다음을 생각할 수 있다.

- routing score를 KV cache eviction 정책에 활용할 수 있을까?
- 오랫동안 선택되지 않는 block을 더 낮은 우선순위로 둘 수 있을까?
- prefix cache에서 자주 재사용되는 block과 MoBA routing 정보를 함께 사용할 수 있을까?

이것들은 공개 자료에서 검증된 기능이라기보다, MoBA-style routing을 serving system에 확장해볼 수 있는 연구 방향에 가깝다.

### long-context RAG

RAG에서는 검색된 문서 조각이 길게 붙는 경우가 많다. 이때 모든 문서 조각이 모든 query token에 똑같이 중요하지는 않다.

MoBA-style block routing은 문서 chunk 단위 relevance와 잘 맞을 가능성이 있다.

```text
RAG prompt

[system] [question] [doc chunk 1] [doc chunk 2] [doc chunk 3] ...
                              |
                              v
                     query별 관련 chunk/block 선택
```

다만 실제 RAG 품질은 검색기, chunking, prompt format, model training에 강하게 영향을 받는다. 따라서 "MoBA를 쓰면 RAG가 좋아진다"라고 단정할 수는 없다.

### repo-level code analysis

코드베이스 분석도 비슷하다. repository는 자연스럽게 file, class, function 단위 block 구조를 가진다. 그래서 block sparse routing과 잘 어울릴 수 있다.

```text
Repository context

[README] [api.py] [models.py] [tests/] [config/]
                       |
                       v
              query와 관련 있는 file/function block 선택
```

예를 들어 "이 함수가 깨지는 이유"를 묻는 query는 전체 repo보다 관련 파일, 호출 경로, 테스트 block에 더 집중하는 편이 효율적일 수 있다.

물론 이것도 연구 아이디어다. 공개 자료 기준으로 MoBA가 repo-level code analysis에 대해 어떤 성능을 보이는지는 단정하기 어렵다.

## 내가 이해한 핵심

MoBA는 long-context LLM에서 full attention의 quadratic bottleneck을 줄이려는 block sparse attention 구조다. Linear attention처럼 attention 계산식을 크게 바꾸기보다는, 기존 attention의 형태를 유지하면서 query별로 볼 block을 줄이는 쪽에 가깝다.

핵심은 간단하다.

```text
모든 token을 다 보지 말고,
block 대표 key로 먼저 관련 block을 고른 뒤,
선택된 block 내부 token에만 attention하자.
```

하지만 이 간단한 아이디어를 실제 모델과 serving system에 넣는 일은 단순하지 않다. top-k, block size, causal masking, current block 포함, kernel 구현, KV cache layout, continue training이 모두 얽힌다.

그래서 MoBA는 "바로 가져다 쓰는 inference trick"이라기보다, long-context LLM을 위한 attention architecture 설계라고 보는 편이 정확하다.

## 한 줄 요약

MoBA는 long-context LLM에서 sequence를 block으로 나누고 query별 top-k block만 선택해 attention함으로써, full attention의 비용을 줄이려는 trainable block sparse attention 구조다.

## 핵심 요약

- Full attention은 모든 query가 모든 key/value를 보는 구조라 long-context에서 `O(n^2)` 계산량과 memory 병목이 커진다.
- Sparse attention은 일부 token이나 block만 보는 방식이고, Linear attention은 attention 계산식 자체를 바꾸는 방식이다.
- MoBA는 context를 block으로 나누고, block 대표 key와 query의 score로 top-k block을 선택한 뒤, 선택된 block 내부 token에 attention한다.
- MoBA는 pretrained model에 attention만 바꿔 바로 쓰는 drop-in inference trick이 아니며, continue training이 필요하다.
- Serving 관점에서는 prefill latency, KV cache, prefix cache, long-context RAG, repo-level code analysis와 연결해볼 만하지만, 실제 통합과 성능은 workload와 구현에 따라 달라진다.

## 참고 자료

- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)
- [MoonshotAI/MoBA 공식 GitHub](https://github.com/MoonshotAI/MoBA)
