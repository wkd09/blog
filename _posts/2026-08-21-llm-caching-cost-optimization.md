---
title: "LLM Caching으로 비용 줄이기: Prompt Cache와 Semantic Cache 설계"
date: 2026-08-21 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
summary: "LLM 캐싱을 exact match, provider prompt cache, semantic response cache로 나누어 보고, 실제 비용 절감률을 계산하는 방법과 Redis 기반 운영 설계를 정리한다."
categories:
  - engineering
tags:
  - AI
  - LLM
  - LLMOps
  - Caching
  - Semantic Cache
  - Prompt Caching
  - Redis
source: "Redis LangCache, Redis Semantic Cache Documentation, OpenAI Prompt Caching Documentation, AI Sparkup, NextGen AI"
---

LLM API를 사용하는 서비스에서는 비슷한 질문에 비슷한 답을 반복해서 생성하는 일이 많다.

FAQ 챗봇에 "비밀번호를 어떻게 바꾸나요?"라는 질문이 들어온 뒤, 다른 사용자가 "로그인 암호 변경 방법을 알려주세요"라고 물어볼 수 있다. 두 요청을 매번 LLM에 보내면 같은 일을 다시 수행하면서 입력 토큰, 출력 토큰, 응답 시간을 모두 지불한다.

![반복되는 LLM 호출로 API 비용이 낭비되는 상황을 표현한 그림]({{ '/assets/images/blog/llm-repeated-call-cost.png' | relative_url }}){: .align-center width="708" }

*같은 의미의 요청도 문자열이 다르면 전통적인 exact-match cache를 통과한다. 이미지: @rauljuncoV.*

캐싱은 이 중복 작업을 재사용하는 방법이다. 다만 LLM 캐싱은 한 가지 기술을 뜻하지 않는다. 정확히 같은 요청의 완성된 답을 재사용할 수도 있고, 공통 prompt의 계산만 재사용할 수도 있으며, 표현이 달라도 의미가 같은 질문의 답을 재사용할 수도 있다.

이 글에서는 LLM 캐싱을 다음 세 종류로 구분하고, 그중 **Semantic Response Cache**를 운영 환경에 적용할 때 무엇을 고려해야 하는지 정리한다.

```text
1. Exact-match response cache : 같은 요청이면 완성된 답변 재사용
2. Provider prompt cache      : 같은 prompt prefix의 계산 재사용
3. Semantic response cache    : 의미가 같은 질문이면 완성된 답변 재사용
```

핵심은 "캐시를 붙이면 비용이 90% 줄어든다"가 아니다. **어떤 계산을 재사용하는지, 실제 cache hit rate가 얼마인지, 잘못된 hit를 어떻게 막을 것인지**가 먼저다.

## LLM 캐싱은 무엇을 저장하는가

세 방식은 모두 캐시라는 이름을 사용하지만 저장 대상과 hit 조건이 다르다.

| 방식 | 저장하거나 재사용하는 것 | Hit 조건 | Hit 이후 LLM 호출 |
| --- | --- | --- | --- |
| Exact-match response cache | 완성된 응답 | 정규화된 요청과 cache key가 동일 | 생략 |
| Provider prompt cache | Prompt prefix의 내부 계산 결과 | 앞부분 token이 정확히 동일 | 필요 |
| Semantic response cache | 질문과 완성된 응답 | 의미 유사도와 metadata 조건 충족 | 생략 |

### Exact-match Response Cache

가장 단순한 방식은 요청을 hash key로 만들어 완성된 응답을 저장하는 것이다.

```text
key = hash(model + system_prompt + user_prompt + tools + data_version)
value = generated_response
```

동일한 요청이 반복되는 batch 작업, 정형화된 prompt, 재시도 요청에 효과적이다. 문자열을 정규화하면 대소문자나 불필요한 공백 차이도 흡수할 수 있다.

정확히 일치해야 하므로 hit rate는 낮을 수 있지만, 임베딩 계산이 없고 false positive 위험이 가장 작다. 따라서 semantic search부터 시작하기보다 **exact cache를 첫 번째 계층에 두는 편이 안전하고 저렴하다.**

### Provider Prompt Cache

Provider prompt cache는 완성된 답을 저장하지 않는다. 긴 system prompt, tool definition, few-shot example, 공통 문서처럼 여러 요청이 공유하는 **정확한 token prefix의 prefill 계산**을 재사용한다.

```text
Request A: [System][Tools][Shared Document][Question A]
Request B: [System][Tools][Shared Document][Question B]
            └────── exact shared prefix ──────┘
```

Cache hit가 발생해도 모델은 새로운 질문과 output token을 처리한다. 따라서 입력 비용과 TTFT(Time To First Token)는 줄일 수 있지만, LLM 호출 전체와 출력 비용이 사라지는 것은 아니다. 같은 prompt라도 sampling 결과는 달라질 수 있다.

[OpenAI Prompt Caching 문서](https://developers.openai.com/api/docs/guides/prompt-caching)는 exact prefix match를 위해 고정 instruction과 example을 앞에, 사용자별 가변 데이터를 뒤에 배치하도록 안내한다. 실제 hit 여부는 usage의 `cached_tokens`로 확인할 수 있다. 지원 범위, 최소 token 수, 보존 시간과 과금 방식은 모델별로 바뀔 수 있으므로 사용하는 provider의 최신 문서를 기준으로 확인해야 한다.

이 방식은 이전 글에서 살펴본 [RadixAttention]({% post_url 2026-08-11-radix-attention %})의 prefix KV cache와 목적이 비슷하다. 차이는 provider API가 관리하는 캐시인지, 자체 serving runtime에서 요청 간 KV cache를 직접 관리하는지에 있다.

### Semantic Response Cache

Semantic cache는 문자열 대신 임베딩 벡터 사이의 거리를 비교한다.

```text
"비밀번호를 어떻게 바꾸나요?"
"로그인 암호 변경 방법을 알려주세요"
                  ↓ embedding
        의미 공간에서 가까운 vector
                  ↓
          기존 답변을 그대로 반환
```

새 질문과 충분히 가까운 과거 질문을 찾으면 LLM을 호출하지 않고 저장된 응답을 반환한다. 표현이 다른 FAQ까지 hit로 만들 수 있어 exact cache보다 높은 hit rate를 기대할 수 있다.

반면 "비슷한 질문"과 "같은 답을 반환해도 되는 질문"은 같은 개념이 아니다. Semantic cache의 가장 큰 문제는 검색 속도가 아니라 **잘못된 답을 재사용하는 false positive**다.

## Semantic Cache의 요청 처리 흐름

기본적인 cache-aside 흐름은 다음과 같다.

![Redis LangCache에서 semantic match 여부에 따라 cache hit와 LLM 호출로 분기하는 흐름]({{ '/assets/images/blog/redis-langcache-flow.png' | relative_url }}){: .align-center width="480" }

*Semantic match가 있으면 저장된 응답을 반환하고, 없으면 LLM을 호출한 뒤 새 응답을 cache에 저장한다. 이미지: @rauljuncoV.*

```text
User Request
    │
    ▼
Normalize + build namespace
    │
    ├─ Exact cache hit ──────────────▶ Return cached response
    │
    ▼
Create query embedding
    │
    ▼
Metadata filter + vector search
    │
    ├─ Similarity above threshold ───▶ Return cached response
    │
    ▼
Provider prompt cache / LLM call
    │
    ▼
Validate response ─▶ Store with embedding, metadata and TTL
```

Semantic cache entry에는 prompt와 response만 저장하면 부족하다. 최소한 다음 정보가 함께 있어야 한다.

```text
prompt
prompt_embedding
response
model_id
system_prompt_hash
tool_schema_hash
knowledge_base_version
tenant_id
locale
safety_policy_version
created_at
ttl
```

예를 들어 model이나 system prompt를 바꾸었는데 예전 답을 그대로 반환하면 새 설정이 반영되지 않는다. RAG 문서를 갱신한 뒤에도 이전 답변이 남아 있으면 cache가 오래된 지식을 전달한다. 그래서 vector similarity를 계산하기 전에 `tenant_id`, `locale`, `knowledge_base_version` 같은 **hard boundary로 검색 범위를 먼저 제한**해야 한다.

[Redis의 Semantic Cache 문서](https://redis.io/docs/latest/develop/use-cases/semantic-cache/)도 prompt, embedding, response와 함께 tenant, locale, model version, safety flag 등의 metadata를 저장하고, vector search와 metadata filter를 함께 적용하는 구조를 설명한다.

## 비용 절감률은 Hit Rate로 계산한다

"최대 90% 절감" 같은 숫자는 모든 서비스에 그대로 적용되는 상수가 아니다. 반복 질문이 얼마나 많은지, cache hit가 전체 LLM 호출을 얼마나 대체하는지에 따라 달라진다.

캐시가 없을 때 월 비용을 다음과 같이 둘 수 있다.

$$
C_{base} = N \times C_{llm}
$$

- $N$: 월 요청 수
- $C_{llm}$: 요청 하나의 평균 LLM 비용

Semantic cache의 hit rate를 $H$, 요청당 embedding과 조회 비용을 $C_{lookup}$, cache miss의 저장 비용을 $C_{write}$라고 하면 단순화한 월 비용은 다음과 같다.

$$
C_{cache} = N \times \left((1-H)C_{llm} + C_{lookup} + (1-H)C_{write}\right)
$$

절감률은 다음처럼 계산할 수 있다.

$$
Savings\ Rate = 1 - \frac{C_{cache}}{C_{base}}
$$

예를 들어 월 100만 요청, 요청당 평균 LLM 비용이 10원, semantic hit rate가 60%, 요청당 cache 관련 비용이 평균 0.2원이라고 가정해 보자.

| 항목 | 비용 |
| --- | ---: |
| 캐시 적용 전 | 1,000,000 × 10원 = 1,000만원 |
| Cache miss의 LLM 비용 | 1,000,000 × 40% × 10원 = 400만원 |
| Embedding·조회·저장 비용 | 약 20만원 |
| 캐시 적용 후 | 약 420만원 |
| 추정 절감률 | 약 58% |

Hit rate가 90%에 가깝고 lookup 비용이 매우 작아야 90%에 가까운 절감이 가능하다. 반대로 질문이 대부분 고유하거나 사용자 context가 매번 다르면 semantic cache를 운영하는 비용이 절감액보다 커질 수도 있다.

Provider prompt cache는 계산식이 다르다. 전체 응답을 건너뛰지 않고 **재사용된 input prefix token에만 할인이나 계산 절감이 적용**되기 때문이다. 따라서 input과 output 비용을 분리해 계산해야 한다.

$$
C_{prompt\ cache} = C_{cached\ input} + C_{uncached\ input} + C_{output}
$$

두 캐시는 경쟁 관계가 아니다. Semantic cache miss에서 LLM을 호출해야 할 때 provider prompt cache가 공통 prefix 비용을 다시 줄일 수 있다.

## Similarity Threshold가 정확도를 결정한다

두 vector의 cosine similarity를 사용한다면 다음과 같이 표현할 수 있다.

$$
similarity(q, k) = \frac{q \cdot k}{\lVert q \rVert \lVert k \rVert}
$$

유사도가 threshold 이상이면 hit로 처리한다.

```python
candidate = semantic_cache.search(
    embedding=embed(normalized_query),
    filters={
        "tenant_id": tenant_id,
        "locale": locale,
        "knowledge_base_version": kb_version,
    },
    top_k=3,
)

if candidate and candidate.similarity >= threshold:
    return candidate.response

response = call_llm(request)
semantic_cache.store(request, response, ttl=ttl)
return response
```

이 코드는 구조를 보여 주기 위한 pseudocode다. 실제 API에서 similarity와 distance 중 어느 값을 쓰는지, 값의 범위와 방향은 embedding model과 vector store마다 확인해야 한다.

Threshold가 너무 높으면 안전하지만 paraphrase를 놓쳐 hit rate가 낮아진다. 너무 낮으면 hit rate는 높아지지만 다른 질문의 답을 반환할 수 있다. Redis는 [Semantic Cache 최적화 가이드](https://redis.io/blog/10-techniques-for-semantic-cache-optimization/)에서 높은 threshold로 시작한 뒤 평가 데이터로 점진적으로 조정할 것을 권한다.

운영 threshold는 감으로 정하지 않고 labeled pair로 검증하는 편이 좋다.

| Query A | Query B | 같은 답을 재사용해도 되는가 |
| --- | --- | --- |
| 비밀번호 변경 방법 | 로그인 암호 바꾸는 법 | Yes |
| 비밀번호 변경 방법 | 비밀번호를 잊었을 때 복구 방법 | 정책에 따라 다름 |
| 서울 날씨 알려줘 | 부산 날씨 알려줘 | No |
| 오늘 환율 알려줘 | 어제 환율 알려줘 | No |

이 데이터에서 threshold별 precision, recall, false positive rate를 측정한다. 일반 검색이라면 관련 결과 몇 개 중 하나가 조금 틀려도 다음 단계에서 보정할 수 있지만, response cache는 검색 결과를 **최종 답으로 바로 노출**한다. 따라서 보통 recall보다 precision을 우선한다.

## 날씨와 주가는 Semantic Cache의 쉬운 예제가 아니다

Semantic cache 설명에서는 "오늘 서울 날씨 어때?"와 "서울에 지금 비가 오나요?"가 자주 예제로 등장한다. 문장의 의미가 가깝다는 설명에는 편리하지만, 실제 response cache 사례로는 위험하다.

같은 질문도 요청 시각에 따라 정답이 바뀌기 때문이다. 사용자 위치, timezone, 조회 시각이 다르면 더욱 그렇다.

시간에 민감한 요청은 다음 중 하나로 처리해야 한다.

- Semantic response cache를 우회한다.
- 시간 구간, 지역, 데이터 version을 hard filter에 포함한다.
- 매우 짧은 TTL을 사용한다.
- 완성된 자연어 답보다 upstream API 결과나 retrieval 결과를 캐시한다.

같은 이유로 주가, 재고, 배송 상태, 계정 잔액, 최신 뉴스는 FAQ보다 훨씬 보수적으로 다뤄야 한다.

## Redis로 구현할 때의 선택지

Redis를 사용한다면 크게 두 가지 선택지가 있다.

![Redis 로고]({{ '/assets/images/blog/redis-logo.png' | relative_url }}){: .align-center width="760" }

| 선택지 | 특징 | 적합한 경우 |
| --- | --- | --- |
| 직접 구성 | Redis Hash/JSON, vector index, TTL, filter를 직접 설계 | 기존 Redis 운영 경험이 있고 세밀한 제어가 필요할 때 |
| Redis LangCache | REST API/SDK로 사용하는 managed semantic cache | embedding, index와 관측 도구 운영을 줄이고 싶을 때 |

직접 구성할 때는 prompt, embedding, response, metadata를 Hash나 JSON에 저장하고 vector field와 metadata field를 함께 index한다. Hit 시 응답을 반환하고, miss 시 LLM 응답을 생성한 뒤 TTL과 함께 저장한다. LRU/LFU eviction은 제한된 memory 안에서 오래 사용하지 않은 entry를 제거하는 데 활용할 수 있다.

[Redis LangCache](https://redis.io/langcache/)는 이 흐름을 managed service로 제공한다. [LangCache API 문서](https://redis.io/docs/latest/develop/ai/context-engine/langcache/api-examples/)의 기본 사용 방식도 LLM 호출 전에 유사 응답을 검색하고, miss이면 LLM 결과를 저장하는 read-through 흐름이다.

다만 제품 페이지의 "최대 90%" 같은 수치는 특정 workload와 조건에서 얻은 vendor claim으로 보는 것이 정확하다. 도입 판단은 자사 traffic으로 측정한 hit rate, false positive rate, cache 비용과 latency를 기준으로 해야 한다.

## 운영 환경에서 꼭 필요한 정책

### 1. Cache Namespace를 분리한다

Cache key나 metadata에 적어도 다음 version을 반영한다.

- model과 model revision
- system prompt
- tool definition과 structured output schema
- RAG knowledge base
- safety policy
- tenant와 locale

설정이 바뀌었을 때 전체 key를 순회하며 지우기보다 version을 올려 새 namespace를 사용하는 편이 단순하다. 이전 namespace는 짧은 기간 뒤 만료시킬 수 있다.

### 2. TTL을 데이터 변화 속도에 맞춘다

모든 entry에 같은 TTL을 적용할 필요는 없다.

| 데이터 | TTL 방향 |
| --- | --- |
| 제품 사용법, 안정된 FAQ | 길게 |
| 자주 수정되는 사내 문서 | 문서 version 연동 |
| 날씨, 재고, 가격 | 매우 짧게 또는 cache 우회 |
| 사용자별 민감 정보 | 공유 cache 금지 또는 강한 격리 |

TTL은 stale response의 최대 생존 시간을 제한한다. 원본 데이터가 갱신될 때 관련 cache를 명시적으로 invalidation하면 더 빠르게 일관성을 맞출 수 있다.

### 3. 생성된 모든 답을 저장하지 않는다

오류 응답, timeout 안내, hallucination 가능성이 높은 답, moderation 실패 응답을 저장하면 같은 문제가 반복된다. Cache write 전에 schema validation, citation 검증, safety check 같은 품질 gate를 둘 수 있다.

고객 지원처럼 정답 집합이 비교적 안정적이면 LLM이 즉석에서 만든 답보다 검수된 canonical response를 미리 넣는 cache warming도 유용하다.

### 4. Multi-turn 대화를 그대로 공유하지 않는다

"그 상품 환불해 줘"처럼 현재 대화에 의존하는 요청은 질문 문장만 embed하면 다른 사용자나 다른 session의 답과 잘못 매칭될 수 있다.

이 경우에는 다음 방법을 고려한다.

- 독립적인 single-turn FAQ에만 semantic cache 적용
- 대화에서 standalone intent와 필요한 slot을 추출한 뒤 key 생성
- user/session scope를 hard filter로 제한
- 개인화가 강한 요청은 cache 우회

### 5. Hit Rate만 보지 않는다

Hit rate를 높이려고 threshold를 낮추면 비용은 줄어도 품질이 무너질 수 있다. 최소한 다음 지표를 함께 본다.

```text
exact_cache_hit_rate
semantic_cache_hit_rate
semantic_false_positive_rate
cache_lookup_p50 / p95
llm_p50 / p95
embedding_cost
cache_storage_cost
avoided_llm_cost
stale_response_rate
```

특히 cache hit 표본을 주기적으로 사람이 평가하거나 LLM judge와 규칙 기반 검사를 조합해, "유사한 질문"이 정말 "같은 답을 가져도 되는 질문"인지 확인해야 한다.

## 권장 도입 순서

처음부터 전체 traffic에 semantic cache를 적용할 필요는 없다.

1. **Traffic을 분석한다.** 정규화한 질문을 clustering해 반복률과 예상 hit rate를 구한다.
2. **Exact cache를 먼저 적용한다.** 동일 요청과 재시도를 저렴하고 안전하게 제거한다.
3. **Prompt 구조를 고정한다.** 공통 instruction, tools, schema를 앞에 두어 provider prompt cache hit를 높인다.
4. **안정적인 FAQ domain을 고른다.** 최신성이나 개인화가 약한 범위부터 semantic cache를 shadow mode로 실행한다.
5. **False positive를 측정한다.** Cache 후보는 찾되 사용자에게 반환하지 않고 실제 LLM 응답과 비교한다.
6. **높은 threshold로 일부 traffic에 적용한다.** 품질을 확인하면서 domain별 threshold와 TTL을 조정한다.
7. **비용과 품질을 함께 평가한다.** 절감한 LLM 비용에서 embedding, storage, network와 운영 비용을 뺀다.

권장 cache 계층은 다음과 같다.

```text
L1  In-process exact cache       가장 빠르고 작은 hot cache
L2  Distributed exact cache      동일 요청의 완성된 응답 재사용
L3  Semantic response cache      paraphrase까지 완성된 응답 재사용
L4  Provider prompt cache        LLM 호출 시 공통 prefix 계산 재사용
L5  LLM inference                최종 cache miss만 처리
```

L3에서 hit가 나면 L4와 L5를 모두 건너뛴다. L3가 miss여도 L4가 공통 prompt의 입력 계산을 줄인다. 이렇게 각 계층이 서로 다른 중복을 제거한다.

## 내가 이해한 핵심

LLM 비용 최적화에서 가장 저렴한 호출은 실행하지 않은 호출이다. Semantic cache는 표현이 다른 반복 질문까지 찾아 완성된 응답을 재사용하므로 FAQ, helpdesk, 안정적인 문서 Q&A에서 큰 효과를 낼 수 있다.

하지만 similarity가 높다는 사실만으로 같은 답을 반환해도 되는 것은 아니다. 실무에서는 다음 네 가지가 hit rate보다 먼저다.

- Version과 tenant를 포함한 namespace 격리
- 데이터 변화 속도에 맞춘 TTL과 invalidation
- 평가 데이터로 검증한 domain별 threshold
- False positive와 실제 순절감액을 포함한 관측

따라서 "Semantic Cache로 90% 절감"은 출발점이 아니라 검증할 가설이다. Exact cache, provider prompt cache, semantic response cache를 workload에 맞게 계층화하고, 실제 traffic에서 비용과 품질을 함께 측정해야 한다.

## 참고 자료

- [Redis LangCache](https://redis.io/langcache/)
- [Redis Semantic Cache 공식 문서](https://redis.io/docs/latest/develop/use-cases/semantic-cache/)
- [Redis: 10 techniques to optimize your semantic cache](https://redis.io/blog/10-techniques-for-semantic-cache-optimization/)
- [OpenAI Prompt Caching 공식 문서](https://developers.openai.com/api/docs/guides/prompt-caching)
- [AI Sparkup: Redis 8의 시맨틱 캐싱으로 LLM 비용 90% 절감하기](https://aisparkup.com/posts/4194)
- [NextGen AI: LLM 캐싱 전략](https://trueman1983.tistory.com/entry/LLM-%EC%BA%90%EC%8B%B1Caching-%EC%A0%84%EB%9E%B5-Semantic-Cache%EB%A1%9C-API-%EB%B9%84%EC%9A%A9-90-%EC%A4%84%EC%9D%B4%EA%B8%B0)
