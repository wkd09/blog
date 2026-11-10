---
title: "GraphRAG 논문 정리: 전체 문서 집합을 이해하는 RAG"
date: 2026-07-02 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - RAG
  - GraphRAG
  - KnowledgeGraph
  - Paper
source: "arXiv:2404.16130"
---

# GraphRAG: 전체 문서 집합을 이해하는 RAG

이 글은 논문 [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)을 바탕으로 정리한 글이다.

> Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson  
> arXiv:2404.16130. [[Paper](https://arxiv.org/pdf/2404.16130)]

논문의 핵심 문제의식은 간단하다.

> 일반적인 RAG는 질문과 비슷한 chunk를 찾는 데는 강하지만, 문서 집합 전체를 이해해야 하는 질문에는 약하다.

예를 들어 이런 질문을 생각해보자.

```text
이 뉴스 데이터셋 전체에서 반복적으로 등장하는 공공보건 이슈는 무엇인가?
이 회사의 회의록 전체를 보면 조직의 주요 리스크는 어떻게 변화하고 있는가?
이 논문 묶음에서 AI 윤리와 관련된 주요 주제는 무엇이고, 서로 어떻게 연결되는가?
```

이런 질문은 특정 문장 하나에 답이 들어 있지 않다. 여러 문서에 흩어진 사건, 인물, 조직, 개념, 관계를 모아서 봐야 한다. 논문은 이런 질문을 `global sensemaking question`으로 보고, 이를 **query-focused summarization(QFS)** 문제로 다룬다.

GraphRAG의 한 줄 요약은 다음과 같다.

> GraphRAG는 `문서 -> 지식 그래프 -> 커뮤니티 요약 -> map-reduce 답변`으로 이어지는 global RAG 방식이다.

![GraphRAG architecture]({{ '/assets/images/blog/graphrag-architecture.png' | relative_url }})

*GraphRAG 구현 아키텍처 예시. 핵심은 특정 DB가 아니라, 전체 corpus를 graph와 community summary로 구조화하는 것이다.*

## 1. 왜 Vector RAG만으로는 부족한가?

일반적인 RAG는 보통 다음 흐름을 따른다.

```text
문서 수집
-> chunk 분할
-> embedding 생성
-> vector DB 저장
-> 질문 embedding
-> 유사한 chunk top-k 검색
-> LLM 답변
```

이 방식은 `local question`에 강하다. 예를 들어 "계약서에서 해지 통보 기간은 며칠인가?"처럼 답이 특정 문단에 있는 질문이라면 top-k chunk 검색만으로도 충분히 잘 동작한다.

문제는 질문이 넓어질 때다.

```text
전체 문서에서 가장 중요한 주제는 무엇인가?
여러 문서에 걸쳐 반복되는 갈등 구조는 무엇인가?
이 corpus에서 기술 리더들은 규제를 어떻게 바라보는가?
```

이런 질문은 검색 문제가 아니라 요약 문제에 가깝다. top-k chunk 몇 개만 보면 일부 사례는 잡을 수 있지만, 전체 corpus의 구조는 놓치기 쉽다.

GraphRAG는 이 문제를 query time 검색만으로 풀지 않는다. 대신 **질문이 들어오기 전에 corpus 전체를 graph와 summary로 미리 구조화**한다.

```text
기존 RAG:
질문이 들어온 뒤 관련 chunk를 찾는다.

GraphRAG:
질문이 들어오기 전에 corpus 전체를 구조화한다.
```

## 2. GraphRAG의 전체 파이프라인

논문에서 제안하는 GraphRAG의 흐름은 다음과 같다.

```text
문서
-> chunk
-> entity / relationship / claim 추출
-> knowledge graph 생성
-> community detection
-> community summary 생성
-> 질문별 partial answer 생성
-> final answer 생성
```

여기서 중요한 점은 GraphRAG가 단순히 "그래프 DB를 붙인 RAG"가 아니라는 것이다. Neo4j, NetworkX, igraph 같은 도구를 쓸 수는 있지만, 논문의 핵심은 DB 제품이 아니라 **그래프 기반 요약 인덱스**다.

즉 GraphRAG의 인덱스는 단순한 vector index가 아니다. 문서 안의 개체와 관계를 그래프로 만들고, 그 그래프를 community 단위로 나눈 뒤, 각 community가 무엇을 의미하는지 미리 요약해 둔 구조다.

## 3. Entity, Relationship, Claim 추출

GraphRAG는 각 chunk에서 LLM을 사용해 구조화된 정보를 추출한다.

| 요소 | 의미 | 예시 |
|---|---|---|
| Entity | 사람, 조직, 장소, 개념, 사건 등 | `OpenAI`, `Sam Altman`, `AI regulation` |
| Relationship | entity 사이의 관계 | `Sam Altman - leads - OpenAI` |
| Claim | entity나 관계에 대한 검증 가능한 주장 | `OpenAI released GPT-4 in 2023` |

예를 들어 다음 문장이 있다고 하자.

```text
Quantum Systems acquired NeoChip in 2016 to expand its AI accelerator business.
```

LLM은 여기서 다음 정보를 뽑을 수 있다.

```text
Entity:
- Quantum Systems
- NeoChip
- AI accelerator business

Relationship:
- Quantum Systems acquired NeoChip
- NeoChip is related to AI accelerator business

Claim:
- Quantum Systems acquired NeoChip in 2016.
```

이 단계는 전체 품질을 크게 좌우한다. entity를 잘못 뽑거나 relationship을 느슨하게 만들면, 이후 graph community와 summary도 함께 흔들린다. 그래서 실제 구현에서는 도메인에 맞는 entity schema와 few-shot 예시가 중요하다.

## 4. Knowledge Graph와 Community Summary

추출된 entity와 relationship은 knowledge graph로 저장된다.

```text
Node = entity
Edge = relationship
Edge weight = 같은 관계가 발견된 횟수 또는 강도
Claim = node나 edge에 붙는 근거 정보
```

예를 들어 AI 연구 논문 corpus라면 사람, 논문, 방법론, benchmark, metric 등이 node가 되고, `proposes`, `uses`, `evaluates_on`, `improves` 같은 관계가 edge가 될 수 있다.

![GraphRAG example graph]({{ '/assets/images/blog/graphrag-example-graph.png' | relative_url }})

*지식 그래프 예시. 사람, 주제, 논문, 개념이 관계로 연결되고, 이 연결 구조를 통해 corpus 안의 주제 덩어리를 파악할 수 있다.*

그래프가 만들어지면 GraphRAG는 Leiden 알고리즘으로 community detection을 수행한다. 쉽게 말하면 그래프 안에서 서로 강하게 연결된 주제 덩어리를 찾는 것이다.

예를 들어 뉴스 corpus라면 다음 같은 community가 생길 수 있다.

```text
Community 1:
공공보건, 백신, 병원, 질병관리청, 감염병

Community 2:
반도체, 수출 규제, 공급망, 삼성전자, 대만

Community 3:
교육 정책, 입시, 학교, 교사, 학부모
```

그 다음 각 community의 entity, relationship, claim을 바탕으로 LLM이 summary를 만든다. 이 summary는 특정 문단 요약이 아니라, graph 안의 연결 구조를 요약한 것이다.

```text
이 community는 AI 규제와 기업 거버넌스를 중심으로 형성되어 있다.
주요 entity는 OpenAI, Microsoft, EU AI Act, model safety이며,
핵심 관계는 모델 출시와 규제 감독 강화 사이의 긴장이다.
```

이 community summary가 GraphRAG의 핵심 인덱스다. 원문 전체를 매번 다시 읽지 않고, 미리 만들어 둔 summary를 query time에 재사용할 수 있다.

## 5. Query Time: Map-Reduce 답변

질문이 들어오면 GraphRAG는 여러 community summary를 사용해 partial answer를 만든다. 그 뒤 partial answer들을 다시 합쳐 final answer를 생성한다.

```text
User question
-> community summary 여러 개 사용
-> 각 summary에 대해 partial answer 생성
-> partial answer에 helpfulness score 부여
-> score가 낮은 답변 제거
-> score가 높은 답변을 모아 final answer 생성
```

예를 들어 질문이 다음과 같다고 하자.

```text
이 AI 윤리 corpus에서 반복적으로 등장하는 핵심 쟁점은 무엇인가?
```

그러면 각 community는 서로 다른 관점을 제공할 수 있다.

```text
Community A: 모델 편향과 공정성
Community B: 투명성과 설명 가능성
Community C: 데이터 프라이버시와 동의
Community D: 규제, 거버넌스, 책임 소재
```

최종 답변은 이 partial answer들을 합쳐 corpus 전체의 주요 쟁점을 정리한다. 이것이 GraphRAG가 global question에서 일반 vector RAG보다 강한 이유다.

## 6. 실험과 주요 결과

논문은 두 개의 약 100만 token 규모 corpus에서 GraphRAG를 평가했다.

| 데이터셋 | 규모 | chunk 수 | chunk size |
|---|---:|---:|---:|
| Podcast transcript | 약 100만 token | 1,669개 | 600 token |
| News articles | 약 170만 token | 3,197개 | 600 token |

비교 대상은 크게 세 가지다.

| 방식 | 의미 |
|---|---|
| GraphRAG community summary | graph community level별 요약 사용 |
| TS | source text를 map-reduce로 직접 요약 |
| SS | semantic search 기반 vector RAG baseline |

평가는 LLM-as-a-judge 방식으로 진행했고, 기준은 네 가지다.

| 기준 | 의미 |
|---|---|
| Comprehensiveness | 질문의 여러 측면을 충분히 다루는가 |
| Diversity | 다양한 관점과 insight를 제공하는가 |
| Empowerment | 사용자가 주제를 이해하고 판단하는 데 도움이 되는가 |
| Directness | 질문에 직접적이고 명확하게 답하는가 |

핵심 결과는 명확하다.

> GraphRAG 계열 global approach는 semantic search 기반 vector RAG보다 comprehensiveness와 diversity에서 더 좋은 결과를 보였다.

특히 흥미로운 점은 root-level community summary만 사용해도 semantic search baseline보다 좋은 결과를 보였다는 것이다. 즉 GraphRAG가 항상 query time에 많은 token을 써서 이기는 방식은 아니다. index time에 community summary를 잘 만들어두면, query time에는 압축된 context만으로도 global answer를 만들 수 있다.

## 7. 구현할 때의 포인트

GraphRAG를 처음부터 논문 수준으로 구현하려고 하면 꽤 무겁다. 작게 시작한다면 다음 흐름이면 충분하다.

```text
1. 문서 20~100개를 준비한다.
2. 문서를 500~800 token 단위로 chunking한다.
3. 각 chunk에서 entity와 relationship을 JSON으로 추출한다.
4. entity normalization을 간단히 수행한다.
5. NetworkX나 Neo4j에 graph를 만든다.
6. Louvain 또는 Leiden으로 community를 찾는다.
7. community별 summary를 생성한다.
8. 질문이 들어오면 summary별 partial answer를 만든다.
9. partial answer를 합쳐 final answer를 생성한다.
```

초기에는 graph DB 없이 파일 기반으로도 실험할 수 있다.

```text
chunks.jsonl
entities.jsonl
relationships.jsonl
communities.json
community_summaries.json
```

실무에서 특히 중요한 것은 entity normalization이다. `OpenAI`, `Open AI`, `OpenAI Inc.`, `the company behind ChatGPT`처럼 같은 대상을 다른 이름으로 부르는 경우가 많기 때문이다. 같은 entity를 분리하면 graph가 쪼개지고, 다른 entity를 합치면 graph가 오염된다.

또 데이터 유형에 따라 구현 방식도 달라져야 한다. 뉴스, 회의록, 논문 같은 비정형 텍스트는 LLM 기반 extraction이 유용하지만, CSV나 DB table처럼 이미 schema가 있는 데이터는 규칙 기반 graph construction이 더 정확할 수 있다.

## 8. 한계와 주의점

GraphRAG는 강력하지만 만능은 아니다.

첫째, 인덱싱 비용이 크다. 일반 vector RAG는 embedding을 만들면 되지만, GraphRAG는 entity extraction, relationship extraction, graph construction, community detection, summary 생성이 추가된다.

둘째, 추출 품질에 크게 의존한다. LLM이 entity나 relationship을 잘못 추출하면 graph와 summary가 함께 흔들린다.

셋째, hallucination이 자동으로 해결되는 것은 아니다. GraphRAG는 global question에서 더 포괄적이고 다양한 답변을 만들 수 있지만, 답변의 모든 claim이 원문에 의해 검증된다는 뜻은 아니다. 실제 서비스에서는 source attribution과 claim verification을 함께 설계해야 한다.

넷째, 모든 질문에 GraphRAG가 필요한 것은 아니다. 단순 사실 질문에는 vector RAG가 더 싸고 빠르다.

```text
local fact question -> vector search
global sensemaking question -> GraphRAG
entity relationship question -> graph query
```

## 내가 이해한 핵심

GraphRAG의 핵심은 다음과 같다.

```text
문서 전체에서 entity와 relationship을 추출한다.
그 결과로 knowledge graph를 만든다.
그래프에서 community를 찾는다.
각 community를 미리 요약한다.
질문이 들어오면 community summary들로 partial answer를 만들고,
그 답들을 다시 합쳐 global answer를 만든다.
```

Vector RAG가 local fact retrieval에 강하다면, GraphRAG는 global sensemaking question에 강하다. 다만 비용, entity extraction 품질, entity resolution, hallucination 검증 문제는 여전히 중요하다.

## 한 줄 요약

GraphRAG는 그래프 DB를 붙인 RAG가 아니라, 전체 문서 집합을 이해하기 위해 knowledge graph와 community summary를 만드는 global summarization index다.
