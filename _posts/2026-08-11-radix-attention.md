---
title: "RadixAttention 정리: Radix Tree로 Prefix KV Cache를 재사용하는 방법"
date: 2026-08-11 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
summary: "Prefix caching의 기본 원리를 짧게 살펴보고, SGLang의 RadixAttention이 radix tree와 cache-aware scheduling으로 KV cache를 재사용하는 과정을 정리한다."
categories:
  - engineering
tags:
  - AI
  - LLM
  - Serving
  - Prefix Caching
  - KV Cache
  - RadixAttention
  - SGLang
source: "SGLang paper, LMSYS RadixAttention Blog, SGLang RadixCache implementation, The Synaptic Stack, PyTorchKR"
---

LLM 애플리케이션에서는 여러 요청이 긴 앞부분을 공유하는 경우가 많다.

System prompt, tool definition, few-shot example, 검색된 문서와 이전 대화 기록은 그대로이고 마지막 질문만 달라질 수 있다.

```text
Request A: [System Prompt][Document][Question A]
Request B: [System Prompt][Document][Question B]
            └──── shared prefix ────┘
```

두 요청을 완전히 독립적으로 처리하면 Request B는 이미 Request A에서 계산했던 prompt prefix의 KV cache를 다시 계산한다.

**Prefix caching**은 이 중복 prefill을 줄이는 최적화다. SGLang은 prefix caching을 체계적으로 처리하기 위해 **RadixAttention**을 제안했다. RadixAttention은 token sequence와 KV cache 위치를 radix tree로 관리하고, longest prefix match, incremental insertion, LRU eviction과 cache-aware scheduling을 하나의 runtime 안에서 연결한다.

이 글에서는 prefix caching의 기본 원리를 짧게 정리한 뒤 RadixAttention의 자료 구조와 요청 처리 과정을 자세히 살펴본다.

## Prefix Caching을 먼저 짧게 이해하기

Prefix는 token sequence의 **맨 앞에서부터 연속해서 같은 구간**이다.

```text
Request A: [11, 25, 38, 91, 7]
Request B: [11, 25, 38, 44, 2]
            └── shared prefix ──┘
```

LLM에서는 눈에 보이는 문자열보다 tokenizer를 거친 token ID가 기준이다. 같은 문장처럼 보여도 chat template, 공백, tool 순서가 달라져 token ID가 바뀌면 cache hit가 끊길 수 있다.

### KV Cache와 Prefix Cache

일반적인 KV cache는 **한 요청의 decode step 사이**에서 사용한다. 이전 token의 `Key`, `Value`를 저장하고 다음 token을 생성할 때 재사용한다.

Prefix cache는 이 KV cache의 수명을 요청 바깥으로 확장한다. 이전 요청이 끝난 뒤에도 KV를 남겨두고, 같은 prefix를 가진 다음 요청이 이를 재사용한다.

| 구분 | KV cache | Prefix cache |
| --- | --- | --- |
| 재사용 범위 | 한 요청의 decode step 사이 | 서로 다른 요청 사이 |
| 줄이는 계산 | 이전 token의 반복 K/V 계산 | 공통 prompt의 반복 prefill |
| 주로 좋아지는 지표 | Decode throughput | TTFT, prefill latency와 throughput |

Prefix cache는 완성된 답변을 저장하는 response cache와도 다르다. Cache hit가 발생해도 새로운 suffix와 output token은 모델이 다시 계산한다. Sampling을 사용하면 같은 prompt라도 답변이 달라질 수 있다.

### 같은 Prefix의 KV를 재사용할 수 있는 이유

Causal language model에서 위치 `i`의 hidden state는 자신과 앞선 token에만 의존한다.

$$
h_i = f(x_1, x_2, \dots, x_i)
$$

각 layer의 `Key`, `Value`도 이 hidden state에서 만들어진다.

$$
K_i = h_iW_K, \quad V_i = h_iW_V
$$

두 요청이 위치 `i`까지 정확히 같은 token prefix를 갖고 모델과 adapter 등의 조건도 같다면, 그 위치까지의 KV도 같다. 뒤에 붙는 질문은 앞에서 이미 계산한 causal state에 영향을 주지 않는다.

```text
Request A: [P1 P2 P3 P4][A1 A2]
Request B: [P1 P2 P3 P4][B1 B2]
            └ cached KV ┘ └ 새로 계산
```

따라서 prefix cache hit가 발생하면 공통 prefix의 prefill을 건너뛰고 cache에 없는 suffix만 계산할 수 있다.

### Prefix Caching의 한계

Prefix caching은 새로운 output token을 생성하는 decode를 줄이지 않는다.

$$
T_{total} = T_{prefill} + T_{decode}
$$

긴 prompt와 짧은 output에서는 TTFT가 크게 줄 수 있지만, output이 매우 길어 decode가 대부분을 차지하면 end-to-end latency 개선은 작아질 수 있다. 공통 prefix가 짧거나 cache에서 이미 축출된 경우에도 이득이 없다.

여기까지가 prefix caching의 핵심이다. 이제 이 cache를 실제 serving runtime에서 어떻게 찾고, 합치고, 나누고, 버릴 것인지가 남는다.

## RadixAttention이란 무엇인가

RadixAttention은 SGLang이 제안한 **요청 간 자동 KV cache 재사용 기법**이다.

이름에 `Attention`이 들어가지만 `QK^T`, softmax 같은 attention 수식을 새로 만드는 알고리즘은 아니다. 모델의 출력도 바꾸지 않는다. 기존 attention이 사용하는 KV cache를 radix tree로 관리하고, cache reuse에 유리하도록 요청을 scheduling하는 serving 최적화다.

RadixAttention은 크게 세 부분으로 볼 수 있다.

1. Token sequence와 KV cache 위치를 radix tree로 관리한다.
2. 사용하지 않는 cache를 LRU 정책으로 축출한다.
3. 공통 prefix가 많은 요청을 가까이 실행하도록 cache-aware scheduling을 사용한다.

```text
RadixAttention
├── Radix tree: longest prefix match와 incremental insertion
├── KV memory pool: 실제 KV tensor 저장
├── Lock + LRU: 사용 중인 경로 보호와 leaf eviction
└── Cache-aware scheduler: cache locality가 높은 요청 우선 실행
```

즉 RadixAttention은 단순히 tree 하나를 만드는 기능이 아니라 **prefix cache의 자료 구조와 lifecycle, scheduler를 함께 설계한 방식**이다.

### SGLang과 RadixAttention은 같은 것이 아니다

SGLang은 원 논문을 기준으로 frontend language와 backend runtime을 함께 가리킨다.

```text
SGLang
├── Frontend: LLM program을 표현하는 Python 기반 DSL
│   ├── Prompt와 generation 관리
│   ├── select 같은 decoding 제어
│   └── fork, join을 이용한 parallelism 표현
└── Backend Runtime
    ├── RadixAttention
    ├── Continuous batching
    └── Structured output 최적화
```

Frontend는 여러 LLM 호출과 branch를 하나의 프로그램으로 표현한다. Backend의 RadixAttention은 그 호출들 사이에서 반복되는 prompt state를 발견해 KV cache를 재사용한다.

예를 들어 `fork`로 동일한 prompt state에서 세 개의 평가 branch를 만들면 frontend 관점에서는 병렬 LLM 호출이고, RadixAttention 관점에서는 하나의 공통 prefix에서 세 leaf가 갈라지는 tree다.

```text
[공통 Essay와 평가 지침]
 ├─ [논리성 평가]
 ├─ [문체 평가]
 └─ [사실성 평가]
```

![SGLang의 fork와 gen을 사용한 다차원 에세이 평가 예시]({{ '/assets/images/blog/sglang-essay-judge-example.png' | relative_url }})

*SGLang frontend는 하나의 essay state를 `fork`해 clarity, originality, evidence 관점의 평가를 병렬로 생성한다. RadixAttention은 각 branch가 공유하는 essay와 평가 지침의 KV cache를 backend에서 재사용할 수 있다. 출처: LMSYS SGLang 소개 글 Figure 5.*

따라서 SGLang의 성능을 이해할 때는 프로그램 내부 parallelism과 RadixAttention의 cache reuse를 구분해야 한다. 둘은 함께 사용할 때 시너지가 있지만 서로 같은 최적화는 아니다.

## 왜 Radix Tree를 사용하는가

Prefix cache에는 세 가지 핵심 연산이 필요하다.

- 새로운 요청과 cache 사이의 longest prefix match
- 새로 계산한 token sequence의 incremental insertion
- GPU memory가 부족할 때 재사용 가능성이 낮은 cache의 eviction

일반 hash map에 전체 prompt만 key로 저장하면 완전히 같은 prompt는 찾기 쉽지만 부분 prefix를 찾기 어렵다. 가능한 모든 prefix를 별도 key로 넣으면 index 수가 크게 증가한다.

Tree 구조는 공통 prefix를 하나의 경로로 자연스럽게 공유한다.

### Trie와 Radix Tree의 차이

일반 trie는 edge 하나에 token 하나를 저장한다.

```text
root
 └─ A
    └─ B
       └─ C
          └─ D
```

Token 하나마다 node가 생기므로 분기가 없는 긴 경로에도 많은 node가 필요하다.

Radix tree는 child가 하나뿐인 연속 경로를 하나의 edge로 압축한다. Edge label에 token sequence 전체를 넣을 수 있다.

```text
root
 └─ [A B C D]
```

두 sequence가 중간에서 갈라지면 공통 부분까지만 하나의 edge로 남기고 나머지를 분기한다.

```text
[A B C D E F]
[A B C X Y]

root
 └─ [A B C]
     ├─ [D E F]
     └─ [X Y]
```

이 구조는 LLM workload와 잘 맞는다. 긴 system prompt나 chat history는 공통 경로가 되고, 서로 다른 사용자 질문이나 sampling 결과는 branch가 된다.

## Radix Tree에는 무엇을 저장하는가

개념적으로 RadixAttention은 다음 mapping을 관리한다.

```text
token sequence -> KV cache tensor의 물리적 위치
```

실제 KV tensor는 크기 때문에 GPU의 KV memory pool에 저장된다. CPU에 있는 radix tree가 tensor 전체를 직접 들고 있는 것이 아니라, 해당 token의 KV가 위치한 GPU slot 또는 page index를 가리킨다.

```text
CPU: Radix Tree

root
 └─ key=[A B C]
    value=[slot 8, slot 21, slot 4]

GPU: KV Memory Pool

slot 4  -> token C의 모든 layer KV
slot 8  -> token A의 모든 layer KV
slot 21 -> token B의 모든 layer KV
```

SGLang의 현재 `RadixCache` 구현에서 node는 대략 다음 정보를 가진다.

- `key`: parent에서 현재 node까지의 token ID sequence
- `value`: token에 대응하는 KV cache index
- `children`: 다음으로 분기되는 child node
- `parent`: 부모 node
- `last_access_time`: LRU 판단에 사용하는 접근 시각
- `lock_ref`: 실행 중인 요청이 이 경로를 사용하는지 나타내는 참조 횟수
- `extra_key`: LoRA, cache version처럼 같은 token이라도 cache를 분리해야 하는 namespace

현재 구현은 page size가 1보다 크면 matching key를 page 경계에 맞춘다. 따라서 radix tree가 논리적으로 token sequence를 표현하더라도 실제 재사용 단위는 KV memory layout의 page 단위와 연결된다.

### Extra Key가 필요한 이유

Token ID가 같다고 항상 KV를 공유할 수 있는 것은 아니다. 서로 다른 model revision이나 LoRA adapter를 사용하면 같은 token sequence에서도 KV가 달라진다.

SGLang의 `RadixKey`는 token ID와 선택적인 `extra_key`를 함께 사용한다. Token prefix가 같아도 `extra_key`가 다르면 별도의 namespace로 취급해 같은 node를 공유하지 않는다.

```text
RadixKey(tokens=[A B C], extra_key=adapter_A)
RadixKey(tokens=[A B C], extra_key=adapter_B)
```

이는 cache hit rate보다 정확성과 격리가 우선이라는 뜻이다.

## Radix Tree가 만들어지는 과정

간단한 token 예제로 tree가 어떻게 바뀌는지 살펴보자.

### Step 1. 첫 번째 요청 삽입

첫 요청이 다음 sequence를 처리했다.

```text
Request 1: [A B C D E F]
```

Cache가 비어 있으므로 전체 sequence를 prefill하고 KV 위치와 함께 하나의 edge로 삽입한다.

```text
root
 └─ [A B C D E F]
```

### Step 2. Edge 중간에서 Prefix가 일치하는 요청

두 번째 요청이 들어온다.

```text
Request 2: [A B C X Y]
```

기존 edge `[A B C D E F]`와 비교하면 `[A B C]`까지 일치한다. Match가 edge 중간에서 끝나므로 기존 node를 split한다.

```text
split 전

root
 └─ [A B C D E F]

split 후

root
 └─ [A B C]
     ├─ [D E F]
     └─ [X Y]
```

`[A B C]`의 KV는 두 요청이 공유한다. Request 2는 `[X Y]`만 새로 계산한다.

Node split은 KV tensor를 다시 계산하거나 복제하는 작업이 아니다. 기존 node가 들고 있던 token key와 KV index 구간을 공통 부분과 나머지 부분으로 나누어 tree boundary를 새로 만드는 bookkeeping에 가깝다.

### Step 3. 더 짧은 Prefix에서 다시 분기

세 번째 요청이 `[A B G H]`라면 기존 `[A B C]` edge도 중간에서 split된다.

```text
root
 └─ [A B]
     ├─ [C]
     │   ├─ [D E F]
     │   └─ [X Y]
     └─ [G H]
```

이제 `[A B]`는 세 요청이 공유하고 `[A B C]`는 앞의 두 요청만 공유한다. Tree는 요청이 들어올 때마다 workload에 실제로 나타난 공통 prefix 경계를 자동으로 드러낸다.

### Step 4. 기존 대화의 연장

첫 번째 경로에서 대화가 계속되어 `[I J]`가 붙었다고 해보자.

```text
Request 4: [A B C D E F I J]
```

Runtime은 `[A B] -> [C] -> [D E F]`를 cache에서 찾고 `[I J]`만 계산해 leaf 뒤에 추가한다.

```text
root
 └─ [A B]
     ├─ [C]
     │   ├─ [D E F]
     │   │   └─ [I J]
     │   └─ [X Y]
     └─ [G H]
```

SGLang은 prompt token뿐 아니라 generation result의 KV도 tree에 유지할 수 있다. 그래서 multi-turn chat에서 이전 assistant response까지 포함된 history를 다음 요청의 prefix로 재사용할 수 있다.

## LMSYS의 9단계 예시로 보는 Tree 변화

LMSYS의 RadixAttention 소개 글은 chat, few-shot learning과 self-consistency 요청이 섞인 상황에서 radix tree가 아홉 단계에 걸쳐 변하는 예시를 보여준다. 앞에서 본 token 예제를 실제 LLM workload로 확장한 흐름이다.

![요청 처리에 따라 분기, 재사용, 축출되는 RadixAttention tree의 9단계 변화]({{ '/assets/images/blog/radix-attention-tree-lifecycle.png' | relative_url }})

*초록색 node는 새로 추가된 cache, 파란색 node는 해당 단계에서 다시 사용된 cache, 주황색 점선과 X는 memory pressure로 축출된 경로를 나타낸다. 출처: LMSYS SGLang 소개 글 Figure 4.*

| 단계 | 들어오는 작업 | Radix tree에서 일어나는 일 |
| ---: | --- | --- |
| 1 | 서버 시작 | Tree가 비어 있다. |
| 2 | 첫 chat turn | System prompt, user message와 model response가 하나의 긴 경로로 삽입된다. |
| 3 | 같은 chat의 다음 turn | 첫 turn 전체가 cache hit되고 새 대화만 leaf 뒤에 추가된다. |
| 4 | 동일 system prompt를 쓰는 새 chat | 기존 edge를 system prompt 경계에서 split하고 두 conversation branch를 만든다. |
| 5 | 두 번째 chat 계속 진행 | 새 turn을 추가하고 memory가 부족하면 오래된 leaf를 축출한다. |
| 6 | 별도의 few-shot 요청 | 기존 chat과 공통 prefix가 없으므로 root 아래에 새로운 subtree가 생긴다. |
| 7 | 같은 example을 쓰는 few-shot batch | Example 구간을 공통 node로 분리하고 질문별 branch를 만든다. |
| 8 | 첫 chat에 새 메시지 도착 | 첫 chat 경로를 다시 사용하고 LRU가 된 다른 chat branch를 제거한다. |
| 9 | 같은 질문에서 추가 sample 생성 | Question prefix를 공유하는 self-consistency branch를 만들고 공간 확보를 위해 다른 leaf를 축출한다. |

이 예시에서 중요한 것은 tree가 사전에 고정되어 있지 않다는 점이다. Runtime은 full prompt를 받아 실제 요청들이 공유하는 경계를 발견할 때마다 node를 split한다. 동시에 memory pressure가 생기면 사용하지 않는 leaf를 지워 tree 모양을 계속 바꾼다.

```text
요청 도착
  -> longest prefix match
  -> 필요한 경계에서 split
  -> suffix 계산 및 branch 삽입
  -> memory 부족 시 LRU leaf eviction
```

Chat, few-shot과 self-consistency가 서로 다른 cache 기능을 사용하는 것도 아니다. 모두 **token sequence의 공통 prefix를 tree path로 표현한다**는 하나의 규칙으로 처리한다.

## 요청 하나가 처리되는 전체 흐름

RadixAttention 관점에서 요청 lifecycle을 정리하면 다음과 같다.

### 1. Full Prompt 전달

Frontend는 cache hit 여부를 직접 판단하지 않고 전체 prompt를 runtime에 전달한다. Runtime이 tokenization 결과를 기준으로 자동 matching한다.

### 2. Longest Prefix Match

`match_prefix()`가 root부터 child를 따라가며 가장 긴 cached prefix를 찾는다. 반환값에는 일치한 token 구간의 KV index와 마지막으로 일치한 tree node가 포함된다.

Matching이 node의 edge label 중간에서 끝나면 그 지점을 node boundary로 만들기 위해 tree를 split할 수 있다.

### 3. Matched Path Lock

현재 요청이 cache hit 경로를 사용하는 동안 해당 node와 ancestor path의 `lock_ref`를 증가시킨다. 실행 중인 요청이 참조하는 KV를 eviction하면 잘못된 memory를 읽게 되므로 이 경로는 보호해야 한다.

### 4. Suffix Prefill과 Decode

Cache hit prefix의 KV index를 요청의 block/page mapping에 연결한다. Cache에 없는 suffix만 prefill하고 이후 output token을 decode한다.

### 5. Tree Insert

새로 계산한 token과 KV index를 tree에 삽입한다. 기존 sequence와 부분적으로 겹치면 node를 split하고, 새로운 suffix는 child node로 추가한다.

### 6. Unlock

요청이 끝나면 `lock_ref`를 감소시킨다. Reference가 0이 된 node는 즉시 삭제되지 않고 이후 요청이 재사용할 수 있는 evictable cache로 남는다.

### 7. Eviction

새 요청에 필요한 GPU KV slot이 부족하면 evictable leaf 중 정책상 우선순위가 가장 낮은 node를 제거해 memory를 반환한다.

주요 연산을 표로 정리하면 다음과 같다.

| 연산 | 역할 | Tree 변화 |
| --- | --- | --- |
| `match_prefix` | 가장 긴 cache hit 구간 탐색 | 필요하면 match 경계에서 split |
| `insert` | 새 token과 KV index 등록 | 기존 node split 또는 leaf 추가 |
| `inc_lock_ref` | 사용 중인 경로 보호 | Node와 ancestor를 non-evictable 상태로 변경 |
| `dec_lock_ref` | 요청 종료 후 보호 해제 | Reference가 없으면 evictable 상태로 변경 |
| `evict` | GPU KV memory 확보 | 사용하지 않는 leaf부터 제거 |

## 왜 Leaf부터 Eviction하는가

RadixAttention의 기본 eviction 정책은 LRU다. 가장 오랫동안 사용하지 않은 cache를 먼저 버린다.

하지만 일반적인 flat LRU cache처럼 임의의 entry 하나를 바로 삭제할 수는 없다. Radix tree의 internal node는 여러 child가 의존하는 공통 prefix다.

```text
root
 └─ [System Prompt]
     ├─ [User A]
     └─ [User B]
```

`[System Prompt]`만 지우고 `[User A]`, `[User B]`를 유지하면 child의 KV가 의존하는 앞선 causal state가 사라진다. Tree 경로도 끊어진다.

따라서 사용 중이지 않은 leaf부터 제거한다.

```text
1. Evict [User A]
2. Evict [User B]
3. [System Prompt]가 leaf가 됨
4. 필요하면 [System Prompt]도 evict
```

현재 SGLang 구현은 evictable leaf를 모아 eviction priority에 따라 꺼낸다. Leaf를 삭제한 뒤 parent에게 child가 하나도 없고 lock도 없다면 parent가 새로운 eviction 후보가 된다. 이런 방식으로 아래에서 위로 재귀적으로 memory를 회수한다.

### Locked Node는 Eviction할 수 없다

Continuous batching 환경에서는 cache를 재사용하는 요청과 새 token을 decode하는 요청이 동시에 실행된다. Running request가 참조하는 경로를 보호하기 위해 node와 ancestor에 lock reference를 둔다.

```text
lock_ref > 0  -> protected, eviction 불가
lock_ref = 0  -> evictable candidate
```

이 때문에 eviction 가능 cache 용량과 현재 실행 중인 요청이 점유한 보호 영역을 구분해서 봐야 한다.

## Cache-aware Scheduling

Radix tree가 있어도 요청 순서가 좋지 않으면 cache hit 기회를 놓칠 수 있다.

다음과 같은 요청들이 대기 중이라고 해보자.

```text
A1: [A B C][1]
A2: [A B C][2]
B1: [X Y Z][1]
B2: [X Y Z][2]
```

FCFS 순서가 `A1 -> B1 -> A2 -> B2`라면 두 prefix 집합을 번갈아 처리한다. Cache memory가 작을 때는 `[A B C]`를 다시 사용하기 전에 축출할 수 있다.

```text
FCFS
A1 -> B1 -> A2 -> B2

Cache locality를 고려한 순서
A1 -> A2 -> B1 -> B2
```

같은 subtree의 요청을 연속해서 처리하면 공통 prefix가 cache에 남아 있을 때 재사용하기 쉽다.

SGLang은 radix tree 정보를 이용하는 cache-aware scheduling policy를 제공한다.

### LPM: Longest Prefix Match

현재 cache와 더 긴 prefix가 일치하는 요청을 우선한다. 이미 GPU에 있는 KV를 많이 재사용하므로 당장 필요한 prefill token을 줄일 수 있다.

```text
Request A cache hit: 4K tokens
Request B cache hit: 512 tokens

LPM -> Request A 우선
```

### DFS 계열 Scheduling

같은 prefix subtree에 속한 요청들을 묶어 depth-first 순서에 가깝게 처리한다. 한 subtree의 작업을 마친 뒤 다른 subtree로 이동하므로 cache locality를 높일 수 있다.

SGLang 논문은 offline batch에서 cache 용량이 가장 긴 request 이상이라는 조건 아래 request radix tree를 DFS 순서로 방문하면 최적 cache hit rate를 얻을 수 있음을 설명한다. Longest-shared-prefix-first 순서가 이러한 DFS 순서를 만든다.

현재 구현에는 `lpm`, `dfs-weight` 같은 cache-aware policy가 있다. 다만 prefix matching과 정렬 자체도 CPU overhead를 만들기 때문에 queue가 매우 크거나 shared prefix가 적다면 항상 이득인 것은 아니다.

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --schedule-policy lpm
```

SGLang의 tuning 문서도 shared prefix가 많은 workload에서 `lpm`을 시도하되 scheduling overhead가 추가된다고 설명한다.

### Scheduling과 Routing은 다른 문제다

RadixAttention의 tree는 기본적으로 한 worker의 local KV cache를 나타낸다. 여러 GPU replica가 있다면 scheduler에 도달하기 전 load balancer가 동일한 prefix 요청을 서로 다른 worker로 보내버릴 수 있다.

```text
Request A(prefix=P) -> Worker 1
Request B(prefix=P) -> Worker 2  # Worker 1의 cache를 사용하지 못함
```

따라서 분산 serving에서는 세 단계를 구분해야 한다.

1. **Radix cache**: Worker 내부에서 prefix KV를 저장하고 찾는다.
2. **Cache-aware scheduling**: Worker의 waiting queue에서 실행 순서를 정한다.
3. **Cache-aware routing**: 여러 worker 중 해당 prefix를 보유한 worker로 요청을 보낸다.

RadixAttention만 활성화했다고 multi-replica 환경의 cache locality가 자동으로 보장되는 것은 아니다.

## RadixAttention이 처리하는 Sharing Pattern

Radix tree는 단일 system prompt 공유뿐 아니라 여러 단계로 분기되는 workload를 표현할 수 있다.

![Few-shot, self-consistency, multi-turn chat, tree-of-thought의 KV cache 공유 패턴]({{ '/assets/images/blog/radix-attention-kv-sharing-patterns.png' | relative_url }})

*파란색은 여러 호출에서 재사용할 수 있는 prefix, 초록색은 요청마다 달라지는 입력, 노란색은 각 branch에서 생성된 output을 나타낸다. 서로 다른 네 workload도 공통 prefix와 branch의 조합으로 표현할 수 있다. 출처: SGLang 논문 Figure 3.*

### Few-shot Learning

```text
[Few-shot Examples]
 ├─ [Question A]
 ├─ [Question B]
 └─ [Question C]
```

동일한 example의 KV를 여러 질문이 공유한다.

### Multi-turn Chat

```text
[System]
 └─ [User 1][Assistant 1]
     └─ [User 2][Assistant 2]
         └─ [User 3]
```

새로운 turn은 이전 conversation history 전체를 prefix로 사용한다.

### Self-consistency

```text
[Question]
 ├─ [Sample 1]
 ├─ [Sample 2]
 └─ [Sample 3]
```

하나의 question에서 여러 reasoning sample을 생성할 때 prompt KV를 공유한다.

### Tree-of-Thought와 Agent Branching

```text
[System][Task][Tool History]
 ├─ [Candidate Action A]
 │   ├─ [Observation A1]
 │   └─ [Observation A2]
 └─ [Candidate Action B]
```

Agent workflow 자체가 tree 구조이므로 radix tree가 각 branch의 공통 history를 자연스럽게 표현한다.

### RAG와 동일한 이미지 질의

같은 retrieved document나 image에 여러 질문을 보내는 경우에도 긴 context를 공유할 수 있다. 멀티모달 입력은 이미지 placeholder만 비교하면 안 되므로 실제 image identity를 cache key에 포함해야 한다.

## vLLM Prefix Caching과 무엇이 다른가

Prefix caching은 상위 개념이고 RadixAttention은 SGLang의 구체적인 구현이다. vLLM도 Automatic Prefix Caching을 제공하지만 자료 구조가 다르다.

| 항목 | vLLM Automatic Prefix Caching | SGLang RadixAttention |
| --- | --- | --- |
| 핵심 자료 구조 | Parent hash로 연결된 KV block hash table | 압축된 token radix tree |
| Prefix 표현 | Full block의 hash chain | 공통 token sequence 경로와 branch |
| Match 방식 | 왼쪽부터 block hash 조회 | Tree를 따라 longest prefix match |
| 경계 처리 | 주로 cache block 경계 | Node split, 실제로는 page alignment 적용 |
| Eviction | Free block queue 기반 LRU | Unlocked leaf 기반 LRU/eviction policy |
| Scheduling 결합 | Block reuse가 핵심 | Tree 기반 cache-aware scheduling까지 포함한 설계 |

둘 다 정확히 같은 token prefix의 KV를 재사용하고 attention 결과를 바꾸지 않는다. 현대 vLLM도 복잡하게 분기되는 prefix를 hash chain으로 표현할 수 있으므로 “radix tree만 multi-level sharing이 가능하다”고 이해하면 안 된다.

핵심 차이는 **prefix caching 지원 여부가 아니라 cache identity와 longest prefix를 어떤 자료 구조로 표현하고 scheduler와 어떻게 연결하는가**에 있다.

## RadixAttention 설명에서 자주 생기는 오해

RadixAttention 입문 자료는 직관을 위해 내부 동작을 단순화하는 경우가 있다. 개념을 잡는 데는 도움이 되지만 실제 구현과 구분할 필요가 있다.

### 오해 1. Node 하나가 항상 Token 하나를 저장한다

Node나 edge마다 token 하나만 저장하는 구조는 일반 trie에 가깝다. Radix tree의 핵심은 child가 하나뿐인 경로를 압축해 **하나의 edge에 여러 token을 저장할 수 있다는 것**이다.

```text
Trie:       A -> B -> C -> D
Radix tree: [A B C D]
```

현재 SGLang 구현은 page size도 고려한다. 따라서 실제 split과 match 경계는 논리적인 token prefix뿐 아니라 KV page alignment의 제약을 받는다.

### 오해 2. Attention 수식 자체를 변경한다

RadixAttention은 attention score나 softmax 계산식을 새로운 근사식으로 바꾸지 않는다. 동일한 prefix에서 이미 계산한 KV tensor를 찾아 기존 attention kernel에 연결한다.

즉 FlashAttention처럼 attention kernel의 memory access를 다시 설계하는 기술과도 역할이 다르다.

```text
FlashAttention  -> 한 번의 attention을 어떻게 효율적으로 계산할 것인가
RadixAttention  -> 이전 요청의 어떤 KV를 다음 요청에서 재사용할 것인가
```

### 오해 3. Prefix가 비슷하면 재사용한다

Radix tree가 자연어의 의미적 유사성을 찾는 것은 아니다. Token ID가 root부터 정확히 일치해야 같은 path를 따라갈 수 있다.

```text
"이 문서를 요약해줘"
"이 자료를 요약해줘"
```

두 문장의 의미가 비슷해도 token prefix가 다르면 같은 RadixAttention cache를 사용하지 않는다.

### 오해 4. 논문의 전체 Speedup이 RadixAttention 하나의 효과다

SGLang은 cache reuse 외에도 frontend parallelism, batching과 structured output 최적화를 포함한다. Benchmark headline은 전체 system 결과이므로 RadixAttention만의 성능으로 인용하면 안 된다.

## 성능 결과를 해석할 때 주의할 점

SGLang 논문은 여러 benchmark에서 기존 시스템보다 최대 `6.4x` 높은 throughput과 최대 `3.7x`의 latency 개선을 보고했다. 하지만 이 수치는 RadixAttention 하나만의 성능이 아니다.

![vLLM 대비 SGLang의 workload별 normalized throughput]({{ '/assets/images/blog/sglang-normalized-throughput.png' | relative_url }})

*Workload별 처리량을 SGLang 기준 1로 정규화한 비교다. Shared prefix 구조와 다른 runtime 최적화의 영향이 workload마다 달라 vLLM과의 차이도 일정하지 않다. 이 그래프는 SGLang 전체 system 비교이며 RadixAttention 단독 ablation이 아니다. 출처: SGLang 논문 Figure 5 기반.*

SGLang의 end-to-end 결과에는 다음 최적화가 함께 들어간다.

- RadixAttention의 KV cache reuse
- 프로그램 내부 parallelism
- Structured output을 위한 compressed finite state machine
- Runtime과 frontend의 co-design

따라서 논문의 headline 수치를 “RadixAttention이 항상 6.4배 빠르다”로 해석하면 안 된다.

논문에서 RadixAttention과 직접 관련해 볼 만한 결과는 다음과 같다.

- Workload에 따라 cache hit rate가 `50%`에서 `99%` 범위로 관찰됐다.
- Cache-aware scheduling은 평균적으로 optimal hit rate의 `96%`에 접근했다.
- Prefix 공유가 없는 실험에서 radix tree 관리 overhead는 `0.3%` 미만이었다.
- Multi-turn chat에서는 output이 짧을 때 효과가 컸고, 긴 output에서는 decode가 지배해 speedup이 작았다.

이 결과도 당시 모델, GPU, prompt 분포와 SGLang 버전에 대한 값이다. 실제 서비스에서는 자체 traffic으로 검증해야 한다.

## 언제 효과가 크고 언제 작을까

RadixAttention의 효과는 tree가 얼마나 멋지게 생겼는지가 아니라 **재사용한 KV token이 실제로 얼마나 많은 prefill을 줄였는지**로 판단해야 한다.

효과가 큰 조건은 다음과 같다.

- 긴 system prompt와 tool schema가 반복된다.
- 같은 문서나 이미지에 여러 질문을 보낸다.
- Multi-turn chat과 agent history가 길다.
- Self-consistency나 tree search처럼 하나의 state에서 여러 branch를 생성한다.
- 동일 prefix 요청이 cache에서 축출되기 전에 다시 도착한다.
- Router가 cache affinity를 유지한다.

효과가 작거나 주의가 필요한 조건은 다음과 같다.

- 요청마다 첫 token부터 서로 다르다.
- 공통 prefix가 매우 짧다.
- Output이 길어 decode 시간이 대부분을 차지한다.
- GPU KV memory가 작아 tree leaf가 재사용 전에 계속 축출된다.
- LoRA, model version, multimodal input이 달라 cache namespace를 공유할 수 없다.
- Cache-aware scheduling 때문에 오래 기다리는 요청이 생기거나 fairness가 나빠진다.
- 여러 replica에 요청이 무작위로 분산된다.

## Cache Hit를 높이는 Prompt 구성

RadixAttention도 정확한 token prefix를 기준으로 동작한다. 가장 기본적인 원칙은 **고정된 내용을 앞에, 자주 바뀌는 내용을 뒤에 배치하는 것**이다.

```text
유리한 구조
[고정 system][고정 tools][공통 examples][공통 document][사용자 질문]

불리한 구조
[timestamp][request ID][사용자 정보][고정 system][공통 document]
```

앞쪽에서 token 하나가 달라지면 그 지점 이후는 같은 tree path를 따라갈 수 없다.

실전에서는 다음 항목을 확인하는 것이 좋다.

- System message와 tool schema 순서를 결정론적으로 유지한다.
- JSON key 순서와 whitespace가 매 요청마다 달라지지 않도록 직렬화한다.
- Timestamp, nonce, request ID 같은 동적 값은 가능한 뒤에 둔다.
- Chat template과 tokenizer version을 고정한다.
- Model과 LoRA adapter가 같은 요청끼리 cache를 공유한다.
- Shared prefix가 많은 요청을 같은 worker로 routing한다.

## 직접 확인해볼 실험

SGLang에서는 RadixAttention이 기본적으로 활성화된다. `--disable-radix-cache`를 사용하면 prefix cache를 끈 baseline을 만들 수 있다.

```bash
# RadixAttention 사용
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct

# Prefix cache를 끈 baseline
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --disable-radix-cache
```

두 서버를 같은 조건에서 별도로 실행하고 다음 요청을 비교하면 된다.

| 실험 | 요청 구성 | 확인할 점 |
| --- | --- | --- |
| Cold cache | 처음 보는 긴 prompt | 전체 prefill latency |
| Exact repeat | 완전히 같은 prompt | 최대 prefix hit |
| Shared document | 긴 문서는 같고 질문만 다름 | 공통 문서 KV 재사용 |
| Mid-prefix change | 중간 token 하나 변경 | 변경 지점 전까지만 hit |
| Multi-turn | 이전 대화에 새 turn 추가 | History KV 재사용 |
| Long output | Prefix는 같고 output을 길게 생성 | TTFT와 전체 latency 차이 |
| Small cache | 동시 요청으로 eviction 유도 | LRU와 hit rate 변화 |

제대로 비교하려면 다음 조건을 통제해야 한다.

1. Model loading과 kernel warm-up을 측정에서 제외한다.
2. Cold cache와 warm cache를 나누어 기록한다.
3. 동일한 model, dtype, batch와 sampling 설정을 사용한다.
4. Output token 수를 고정해 decode 시간의 영향을 통제한다.
5. Prefix 길이와 suffix 길이를 단계적으로 바꾼다.
6. 평균뿐 아니라 p50, p95, p99 TTFT를 측정한다.
7. Cache hit token, eviction, evictable/protected cache 크기를 함께 본다.

`--schedule-policy lpm` 실험도 별도로 수행하는 것이 좋다. Radix cache 자체의 효과와 cache-aware scheduling의 효과를 분리해야 어떤 요소가 개선을 만들었는지 알 수 있다.

## 내가 이해한 핵심

Prefix caching은 여러 LLM 요청이 공유하는 앞부분의 KV cache를 재사용해 중복 prefill을 줄이는 최적화다. RadixAttention은 이 아이디어를 SGLang runtime 안에서 radix tree, LRU eviction과 cache-aware scheduling으로 구현한다.

핵심은 다음과 같다.

- RadixAttention은 새로운 attention 수식이 아니라 KV cache 관리 및 scheduling 기법이다.
- Radix tree는 token sequence를 압축된 edge로 저장하고 공통 prefix를 하나의 경로로 공유한다.
- Tree node는 실제 KV tensor보다 GPU KV memory의 slot 또는 page index를 가리킨다.
- 요청이 기존 edge 중간까지만 일치하면 node를 split해 새로운 공유 경계를 만든다.
- Running request가 사용하는 node와 ancestor path는 lock으로 eviction에서 보호한다.
- GPU memory가 필요하면 사용하지 않는 LRU leaf부터 아래에서 위로 축출한다.
- LPM과 DFS 계열 scheduling은 같은 subtree의 요청을 가깝게 실행해 cache locality를 높인다.
- RadixAttention은 prefill과 TTFT를 줄이지만 output token의 decode 자체를 줄이지 않는다.
- 실제 효과는 shared prefix 길이, 요청 순서, KV memory, eviction, scheduling과 routing에 의해 결정된다.

RadixAttention의 중요한 관점은 KV cache를 요청에 딸린 일회성 상태가 아니라 **여러 LLM 호출이 공유할 수 있는 tree-structured cache**로 본다는 것이다. Multi-turn chat과 agent workflow처럼 호출 구조 자체가 tree에 가까워질수록 이 관점이 더 중요해진다.

## 참고 자료

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [NeurIPS 2024 SGLang Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [Radix Attention: How a Data Structure Inspired a Revolution in Efficient Transformer Inference](https://medium.com/the-synaptic-stack/radix-attention-how-a-data-structure-inspired-a-revolution-in-efficient-transformer-inference-b7d5baa1219a)
- [RadixAttention과 SGLang을 활용한 LLM 프로그래밍 혁신 - PyTorchKR](https://discuss.pytorch.kr/t/radixattention-sglang-llm-feat-lmsys/3318)
- [SGLang RadixCache 구현](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [SGLang Schedule Policy 구현](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- [SGLang Server Arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
- [vLLM Automatic Prefix Caching 설계 문서](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
