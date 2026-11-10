---
title: "LLM Overthinking 논문 정리: 모델은 언제 생각을 멈춰야 할까?"
date: 2026-09-02 00:00:00 +0900
last_modified_at: 2026-09-02 00:00:00 +0900
summary: "TRACE가 LLM의 긴 reasoning을 sub-thought와 progression graph로 분해하고, Explorer와 Late Landing 패턴을 통해 overthinking의 원인과 멈춰야 할 시점을 찾는 방법을 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - Reasoning Model
  - Chain of Thought
  - Overthinking
  - Inference
  - TRACE
  - Paper
source: "ACL 2026: 2026.acl-long.773"
---

# LLM은 언제 생각을 멈춰야 할까?

최근 reasoning model의 추론 비용을 줄이는 방법을 생각하면서 한 가지 아이디어를 떠올리고 있었다.

> 모델이 생성한 reasoning의 길이만 제한하지 말고, 사고가 실제로 앞으로 나아가고 있는지를 보고 멈추게 할 수 없을까?

정답 후보가 더 좋아지는지, 같은 답을 반복해서 검증하는지, 이미 방문한 경로를 다시 도는지를 추적하면 단순한 token budget보다 나은 종료 조건을 만들 수 있을 것 같았다. 그런데 비슷한 문제의식과 접근을 훨씬 구체적인 분석 framework로 발전시킨 논문이 최근 ACL 2026에 나왔다.

이 글은 논문 [Do LLMs Really Need 10+ Thoughts for “Find the Time 1000 Days Later”? Towards Structural Understanding of LLM Overthinking](https://aclanthology.org/2026.acl-long.773/)을 바탕으로 정리한 글이다.

> Xinliang Frederick Zhang, Anhad Mohananey, Alexandra Chronopoulou, Pinelopi Papalampidi, Somit Gupta, Tsendsuren Munkhdalai, Lu Wang, Shyam Upadhyay<br>
> ACL 2026. [[Paper](https://aclanthology.org/2026.acl-long.773.pdf)] [[DOI](https://doi.org/10.18653/v1/2026.acl-long.773)]

논문의 핵심 질문은 단순하다.

> reasoning model은 왜 쉬운 문제에서도 오래 생각하며, 어느 시점부터 그 사고를 overthinking이라고 부를 수 있는가?

저자들은 먼저 14개 thinking model과 6개 task domain에서 이 현상을 측정한다. 이어서 `TRACE`라는 분석 framework로 긴 reasoning을 작은 `sub-thought`로 분해하고, 각 생각 사이의 관계를 graph로 재구성한다.

그 결과 open-weight thinking model의 사고 흐름에서 두 가지 대표 패턴을 발견한다.

- 여러 답을 계속 탐색하는 `Explorer`
- 정답에 도착한 뒤에도 검증을 반복하는 `Late Landing`

한 줄로 요약하면 다음과 같다.

> LLM의 overthinking은 단순히 생각이 긴 현상이 아니라, 추가 생각의 한계효용이 사라졌는데도 탐색이나 검증을 계속하는 현상이다.

## 1. 생각을 오래 하면 항상 더 정확해질까?

Reasoning model은 답을 바로 출력하지 않고 중간 사고 과정을 길게 생성한다. 어려운 수학, coding, planning 문제에서는 test-time compute를 늘리는 것이 실제 성능 향상으로 이어질 수 있다.

하지만 모든 질문에 긴 reasoning이 필요한 것은 아니다.

```text
7 + 2는 얼마인가?
주어진 문서에 답이 존재하는가?
특정 날짜에서 1,000일 뒤는 언제인가?
```

이런 질문에서도 모델은 여러 풀이를 시도하고, 방금 구한 답을 의심하고, 같은 계산을 다른 방식으로 반복한다. 계산량과 응답 시간은 늘어나지만 정확도는 거의 오르지 않거나 오히려 떨어질 수 있다.

논문은 먼저 thinking mode와 non-thinking mode를 정면으로 비교한다. 평가 대상은 Qwen3 계열과 DeepSeek-R1 distillation 계열을 포함한 14개 모델이며, task는 다음 두 범주와 6개 dataset으로 구성된다.

| 범주 | Task | Dataset |
| --- | --- | --- |
| Simple Reasoning | 초등 수학 | ASDiv |
| Simple Reasoning | 날짜 계산 | Date Arithmetic |
| Simple Reasoning | 논리 추론 | Zebra Logic |
| Knowledge Recall | 짧은 문맥 독해 | SQuAD 2.0 |
| Knowledge Recall | 긴 문맥에서 사실 찾기 | NIAH |
| Knowledge Recall | 사실 질의응답 | SimpleQA |

실험 결과, 간단한 문제에서 long-thinking model은 유의미한 정확도 향상 없이 non-thinking mode보다 **5~20배 많은 계산**을 사용했다.

특히 knowledge recall처럼 reasoning workload가 거의 없는 task에서는 문제의 난이도와 관계없이 thinking mode의 이점이 미미했다. 이미 문맥에 있는 사실을 꺼내는 문제라면 오래 생각하는 것보다 정확히 찾는 능력이 더 중요하기 때문이다.

## 2. Thinking이 유효한 구간은 생각보다 좁다

논문은 수학과 날짜 계산 문제의 난이도를 단계별로 높여, 긴 reasoning이 언제 도움이 되는지도 분석한다.

### 너무 쉬운 문제

쉬운 문제는 non-thinking mode도 이미 잘 푼다. 이때 추가 reasoning은 정확도를 거의 올리지 못하므로 대부분의 token이 낭비된다.

Qwen3-235B-A22B의 ASDiv 실험을 보면 가장 쉬운 level에서 정확도는 `97.44% → 100%`로 소폭 증가하지만, 평균 출력 길이는 `46.5 → 320.0 words`로 크게 늘어난다.

### 적당히 어려운 문제

난이도가 올라가면 thinking이 실제로 도움이 된다. 같은 모델의 GSM8K 정확도는 non-thinking mode의 `74.75%`에서 thinking mode의 `91.50%`로 상승한다.

그러나 이 성능 향상에는 10배에 가까운 출력 길이가 필요했다. 논문의 `token waste` 관점에서는 추가 계산 중 약 80%가 측정 가능한 성능 향상으로 연결되지 않았다.

### 모델의 표현 능력을 넘어선 문제

더 오래 생각한다고 항상 한계를 돌파하는 것도 아니다. 날짜 계산 실험의 높은 난이도에서는 수백~수천 일을 일 단위로 세고 윤년이나 달력 규칙까지 처리해야 한다.

Qwen3-235B-A22B의 Temporal-L5 결과는 다음과 같다.

| Mode | 정확도 | 평균 출력 길이 |
| --- | ---: | ---: |
| Non-thinking | 8.66 | 323.9 words |
| Thinking | 45.37 | 3,336.0 words |

Thinking mode가 분명 성능을 높이지만 정확도는 약 45%에서 포화된다. 이미 모델이 문제에 필요한 규칙을 안정적으로 표현하고 조작하는 능력의 한계에 부딪혔다면, token을 더 생성하는 것만으로는 그 한계를 넘기 어렵다.

따라서 긴 reasoning이 유효한 구간은 다음처럼 볼 수 있다.

```text
너무 쉬운 문제
-> 이미 풀 수 있으므로 긴 사고가 낭비됨

적당히 어려운 문제
-> 추가 사고가 정답에 도달하는 데 도움을 줌

모델 능력을 넘어선 문제
-> 오래 생각해도 성능이 빠르게 포화됨
```

중요한 것은 겉으로 보이는 `task complexity`만이 아니다. 실제로 정답을 얻기 위해 필요한 `reasoning workload`와 모델이 그 구조를 처리할 수 있는지도 함께 봐야 한다.

## 3. 기존의 길이 기반 정의는 무엇을 놓치는가?

Overthinking을 가장 쉽게 측정하는 방법은 token 수를 세는 것이다.

논문도 benchmark 단계에서는 다음과 같은 길이 기반 정의를 사용한다.

> Non-thinking mode에서도 맞힌 문제에 대해 thinking mode가 성능 향상 없이 추가로 생성한 thought token을 overthinking으로 본다.

이 정의는 모델과 task 사이의 비용 차이를 비교할 때 유용하다. 하지만 실제 사고 과정의 어느 부분이 필요했고, 어디서부터 낭비가 시작됐는지는 알려주지 못한다.

예를 들어 1,000 token짜리 reasoning 두 개가 있다고 하자.

```text
Reasoning A
초반에 정답 발견 -> 다른 답 탐색 -> 원래 정답 폐기 -> 다시 복귀

Reasoning B
여러 번 수정하며 정답에 접근 -> 정답 발견 -> 같은 답을 반복 검증
```

둘의 길이는 같지만 문제가 발생한 방식은 전혀 다르다. A는 과도한 탐색이 문제이고, B는 정답 이후의 과도한 검증이 문제다. 따라서 효율적으로 멈추는 규칙도 달라야 한다.

저자들은 이 차이를 보기 위해 reasoning을 하나의 긴 문자열이 아니라 구조를 가진 사고 흐름으로 분석한다.

## 4. TRACE: Reasoning을 graph로 재구성하기

`TRACE`는 `Thought-process Reconstruction and Automated Clustering Engine`의 약자다. 긴 reasoning을 작은 생각으로 나누고, 각 생각 사이의 관계를 연결해 사고 진행 graph를 만든다.

![TRACE의 네 단계]({{ '/assets/images/blog/llm-overthinking-trace-framework.png' | relative_url }})

*TRACE는 response sampling, thought decomposition과 label inference, progression graph construction, thought pattern induction의 네 단계로 reasoning을 구조화한다. 출처: 논문 Figure 2.*

전체 과정은 네 단계로 구성된다.

### Stage 1. Response Sampling

먼저 여러 thinking model에서 reasoning trace를 수집한다. 구조 분석에는 20B보다 큰 네 모델을 사용한다.

- Qwen3-30B-A3B
- Qwen3-32B
- R1-Distill-Llama-70B
- Qwen3-235B-A22B

### Stage 2. Thought Decomposition & Label Inference

Gemini 2.5 Pro를 judge로 사용해 reasoning을 `sub-thought` 단위로 분해한다. 논문에서 sub-thought는 다음 조건을 만족하는 최소 단위다.

- 그 자체로 이해할 수 있다.
- 하나의 생각으로 완결되어 있다.
- 답 또는 답 후보를 포함한다.

그다음 이전 sub-thought와 어떤 관계인지 label을 붙인다.

| Label | 의미 |
| --- | --- |
| `Initial` | 첫 접근 또는 계획을 제안한다 |
| `Verification` | 앞의 답이 맞는지 확인한다 |
| `Correction` | 앞의 답을 수정한다 |
| `Backtrack` | 이전에 버렸던 경로로 돌아간다 |
| `Branching Out` | 새로운 방법이나 답 후보를 탐색한다 |
| `Sidetrack` | 현재 풀이와 무관한 방향으로 샌다 |
| `Final` | 최종 답을 전달한다 |

자동 분석이 완벽한 것은 아니다. 저자들이 무작위로 뽑은 sub-thought 200개를 사람이 검사했을 때 label이 합리적이라고 판단한 비율은 93%였다.

### Stage 3. Progression Graph Construction

각기 다른 답 후보를 node로 만들고, sub-thought 사이의 관계를 directed edge로 연결한다.

```text
Query
  -> Answer A
       ├─ verify -> Answer A
       ├─ correct -> Answer B
       └─ backtrack -> Answer A
```

같은 답을 다시 검증하면 self-loop가 생긴다. 새로운 답을 탐색하면 오른쪽으로 진행하고, backtrack 뒤 다른 경로를 탐색하면 위쪽 branch로 표현한다.

### Stage 4. Thought Pattern Induction

개별 graph는 noisy할 수 있다. TRACE는 query type, difficulty, distinct answer 수가 비슷한 graph를 묶고, 자주 등장하지 않는 node와 edge를 제거해 대표 사고 패턴을 만든다.

이 과정을 통해 서로 다른 모델에서 반복적으로 나타나는 두 가지 구조가 드러난다.

## 5. Explorer: 정답을 찾고도 다른 길을 계속 탐색한다

![Explorer 사고 진행 패턴]({{ '/assets/images/blog/llm-overthinking-explorer-pattern.png' | relative_url }})

*Explorer에서는 정답일 확률이 여러 중간 node에 분산된다. 정답을 일찍 찾고도 다른 branch를 탐색하거나 이전 결론을 버릴 수 있다. 출처: 논문 Figure 4.*

`Explorer`는 여러 답 후보와 풀이 경로를 넓게 탐색한다. 정답이 사고 과정의 특정 마지막 node에만 있지 않고 여러 중간 node에 분산되어 나타난다.

이 패턴의 장점은 정답을 초반에 발견할 수 있다는 것이다. 서로 독립적인 경로가 같은 답에 도달하면 그 답의 신뢰도도 높아진다.

하지만 탐색을 멈추지 못하면 문제가 생긴다.

```text
정답 A 발견
-> 혹시 다른가? B 탐색
-> 또 다른 방식으로 C 탐색
-> 처음의 A를 의심하고 폐기
-> 여러 경로를 돈 뒤 A로 복귀
```

이미 맞는 답을 찾았는데도 새로운 가능성을 계속 확인하면서 계산을 낭비한다. 심하면 올바른 중간 답을 버리고 최종 오답을 선택할 수도 있다.

논문에서는 평가한 모델 중 가장 큰 Qwen3-235B-A22B가 이 Explorer 성향을 보였다. 저자들의 분석에서는 이 패턴이 개별 prompt보다 모델 자체의 사고 dynamics와 더 관련되어 있었다.

## 6. Late Landing: 정답에 도착한 뒤에도 확인을 반복한다

![Late Landing 사고 진행 패턴]({{ '/assets/images/blog/llm-overthinking-late-landing-pattern.png' | relative_url }})

*Late Landing에서는 정답 확률이 마지막 node에 집중된다. 마지막의 두꺼운 self-loop는 이미 제시한 답을 반복 검증하는 행동을 뜻한다. 출처: 논문 Figure 5.*

`Late Landing`은 비교적 선형적으로 답을 수정하며 정답에 접근한다. 초반 답의 정답 확률은 낮고, 사고가 진행될수록 마지막 답에 정답 확률이 집중된다.

```text
Answer A
-> correction
-> Answer B
-> correction
-> Answer C
-> 정답 발견
-> verify
-> verify
-> verify
```

이 구조에서는 정답까지 가는 전반부 reasoning이 필요하다. 문제는 정답에 수렴한 뒤에도 confidence를 더 높이기 위해 같은 결론을 반복해서 검증한다는 것이다.

R1-Distill-Llama-70B, Qwen3-30B-A3B, Qwen3-32B 등 논문에서 분석한 대다수 open-weight thinking model이 이 over-verification 패턴을 보였다.

두 패턴을 비교하면 다음과 같다.

| 패턴 | 사고 흐름 | 정답이 나타나는 위치 | 주된 낭비 |
| --- | --- | --- | --- |
| Explorer | 여러 branch를 오가며 탐색 | 초반을 포함한 여러 node | Over-exploration, backtracking |
| Late Landing | 수정하며 선형적으로 수렴 | 마지막 node에 집중 | 정답 이후 over-verification |

## 7. 길이가 아니라 한계효용으로 Overthinking 정의하기

이 논문의 가장 흥미로운 부분은 overthinking의 정의를 바꾼다는 점이다.

$$
\frac{\Delta \text{Performance}}{\Delta \text{Thought}} < \epsilon
$$

> 새로운 sub-thought 하나를 추가했을 때 얻는 성능 향상이 threshold $\epsilon$보다 낮아진 뒤에도 reasoning을 계속하는 것을 overthinking으로 정의한다.

여기서 성능 향상이 급격히 줄어드는 지점을 `convergence point`라고 부른다. 이 정의에서는 reasoning이 500 token인지 5,000 token인지는 본질이 아니다. 추가 사고가 여전히 답을 개선하고 있는지가 중요하다.

![Explorer와 Late Landing의 utility 변화]({{ '/assets/images/blog/llm-overthinking-utility-tracing.png' | relative_url }})

*Explorer는 성능이 초반에 빠르게 오른 뒤 흔들리고, Late Landing은 꾸준히 상승한 뒤 plateau에 도달한다. 초록색 점선은 개입하지 않은 standard thinking mode의 성능이다. 출처: 논문 Figure 6.*

Utility curve에서도 두 패턴의 차이가 드러난다.

- Explorer는 초반에 성능이 빠르게 오르지만 이후 변동이 크다. 추가 탐색이 성능을 떨어뜨리는 구간도 있다.
- Late Landing은 정답을 향해 꾸준히 상승하다 어느 순간 plateau에 도달한다. 이후 sub-thought는 대부분 중복 검증이다.

Temporal-L3 case study에서는 두 모델 모두 8번째 sub-thought가 끝난 시점을 convergence point로 잡았다. 그 뒤 생각 하나를 더 추가했을 때 Qwen3-235B-A22B의 성능은 `63.25 → 62.05`로 하락했고, Qwen3-32B는 `84.76 → 85.06`으로 거의 변하지 않았다.

즉 “몇 token까지 생각할 것인가?”보다 “지금 추가한 생각이 이전 상태를 실제로 개선했는가?”가 더 직접적인 질문이다.

## 8. 정답을 모르는 실전 추론에서는 어떻게 멈출까?

실제 inference에서는 ground truth가 없으므로 $\Delta \text{Performance}$를 직접 계산할 수 없다. 논문은 progression graph에서 관찰되는 행동을 proxy로 사용하는 두 heuristic을 제안한다.

### Self-looping

모델이 답을 제안한 뒤 같은 답을 `k`번 연속으로 검증하면 생성을 종료한다.

```text
Answer A
-> A가 맞는지 검증
-> 다시 A 검증
-> stop
```

Late Landing처럼 정답 이후 확인을 반복하는 모델에 적합하다. 다만 너무 일찍 멈추면 필요한 검증까지 잘라낼 수 있으므로 `k`는 모델 특성에 맞춰 조정해야 한다.

### Backtrack

모델이 다른 경로를 탐색하다 이전에 제안한 답으로 다시 돌아오면 종료한다.

```text
Answer A
-> B 탐색
-> C 탐색
-> A로 복귀
-> stop
```

Explorer 관점에서는 독립적인 탐색 경로가 이전 답을 다시 지지했으므로, 더 탐색하기보다 그 답을 채택할 수 있다는 의미다.

Temporal-L3 실험에서 self-loop 2회 조건만 사용해도 두 모델의 평균 생성 길이는 대략 절반으로 줄었다.

| Model | 설정 | 정확도 | 평균 길이 |
| --- | --- | ---: | ---: |
| Qwen3-235B-A22B | Standard Thinking | 52.87 | 2,722 |
| Qwen3-235B-A22B | Self-loop 2회 | 62.24 | 1,315 |
| Qwen3-235B-A22B | Self-loop 2회 + Backtrack | 62.24 | 1,100 |
| Qwen3-32B | Standard Thinking | 83.84 | 4,000 |
| Qwen3-32B | Self-loop 3회 | 80.18 | 2,463 |

Explorer형 Qwen3-235B-A22B에는 backtrack 조건을 함께 사용하는 것이 효과적이었고, Late Landing형 Qwen3-32B에는 충분한 검증을 허용하도록 self-loop 기준을 3회로 높이는 편이 나았다.

하나의 고정된 token limit가 모든 모델에 최적일 수 없는 이유다. 사고 패턴이 다르면 적절한 종료 신호도 달라진다.

## 9. 이 논문에서 특히 흥미로웠던 점

처음 생각했던 아이디어는 reasoning token을 동적으로 줄이는 것이었다. 예를 들어 답의 변화가 멈추거나 같은 검증이 반복되면 model generation을 조기에 종료하는 방식이다.

이 논문을 읽고 나니 문제를 더 명확하게 나눠 볼 수 있었다.

```text
문제 난이도 예측
-> 처음에 얼마만큼 생각할지 정한다

사고 진행 상태 추적
-> reasoning 중 계속할 가치가 있는지 판단한다

모델별 사고 패턴 파악
-> Explorer와 Late Landing에 서로 다른 종료 규칙을 적용한다
```

특히 `reasoning budget allocation`과 `early stopping`은 다른 문제다. 쉬운 문제를 미리 알아내 적은 budget을 주는 것도 중요하지만, 어려운 문제라도 모델이 정답에 수렴한 순간에는 멈춰야 한다. 반대로 처음 예상보다 어려운 문제라면 budget을 더 줄 수도 있어야 한다.

TRACE는 당장 production runtime에 그대로 넣기에는 무겁다. Gemini 2.5 Pro를 사용해 sub-thought를 분해하고 관계를 추론하므로, online inference마다 같은 분석을 수행하면 오히려 비용이 커질 수 있다.

하지만 offline에서 모델별 사고 패턴을 분석하고, online에서는 더 가벼운 signal을 사용하는 방향은 충분히 생각해볼 수 있다.

- 같은 answer candidate가 반복되는가?
- verification 표현이 연속해서 나타나는가?
- 이전 answer로 backtrack했는가?
- 새로운 sub-thought가 답의 confidence나 consistency를 개선하는가?
- 현재 task의 예상 reasoning workload에 비해 너무 오래 생성하고 있는가?

결국 실용적인 시스템은 `고정 max token`과 `무제한 reasoning` 사이 어딘가에 있을 가능성이 높다. Model이 자신의 진행 상태를 드러내고, runtime이 그 구조를 읽어 budget을 동적으로 조절하는 방식이다.

## 10. 논문의 한계

결과를 해석할 때 몇 가지 범위를 분명히 해야 한다.

첫째, 실험은 의도적으로 비교적 단순하고 답이 명확한 query에 집중한다. 창의적 글쓰기, 복잡한 coding, 장기 planning처럼 여러 경로를 탐색하는 행위 자체가 가치 있는 task에서도 같은 패턴이 일반화되는지는 추가 검증이 필요하다.

둘째, 분석 대상은 공개 reasoning trace를 제공하는 open-weight model 중심이다. 논문이 발견한 Explorer와 Late Landing이 모든 reasoning model에 그대로 적용된다고 단정할 수는 없다.

셋째, sub-thought 분해와 관계 label은 다른 LLM의 판단에 의존한다. 사람의 표본 검사에서 93%가 합리적이었다고 해도, judge model의 bias와 parsing error가 완전히 사라지는 것은 아니다. 실제 case study에서도 Gemini 2.5 Pro의 parsing error로 44개 sample이 제외됐다.

넷째, 제안된 stopping heuristic은 Temporal-L3 case study에서 검증됐다. Model과 task가 달라지면 self-loop 횟수 `k`나 convergence threshold $\epsilon$을 다시 조정해야 한다.

마지막으로 논문이 말하는 `inner workings`는 hidden state나 neural circuit을 직접 분석한 것이 아니다. 모델이 외부로 생성한 reasoning trace의 담화 구조를 분석한 것이다. 따라서 TRACE는 모델 내부 계산 전체라기보다 **표현된 사고 과정의 구조**를 보여주는 도구로 이해하는 편이 정확하다.

## 마치며

이 논문이 주는 가장 중요한 메시지는 “짧게 답하라”가 아니다.

> 필요한 만큼은 생각하되, 더 생각하는 것이 답을 개선하지 못하는 순간을 찾아야 한다.

쉬운 문제에서는 reasoning 자체를 거의 사용하지 않는 편이 낫다. 적당히 어려운 문제에서는 충분히 생각해야 한다. 하지만 모델의 능력 한계에 도달했거나 이미 정답에 수렴했다면, 남은 token은 지능이 아니라 반복일 수 있다.

지금까지 많은 inference optimization이 KV cache, quantization, batching처럼 **token 하나를 싸게 만드는 방법**에 집중했다면, overthinking 연구는 한 단계 앞의 질문을 던진다.

```text
이 token을 더 싸게 생성할 수 있는가?
                 ↓
이 token을 애초에 생성할 필요가 있는가?
```

Reasoning model이 더 널리 사용될수록 두 번째 질문의 중요성은 더 커질 것이다. 긴 사고를 무조건 줄이는 것이 아니라, 사고의 구조와 효용을 읽고 적절한 순간에 멈추는 것. TRACE는 그 방향을 구체적인 graph와 측정 가능한 기준으로 보여준다는 점에서 흥미로운 논문이다.

## 참고 자료

- [ACL Anthology 논문 페이지](https://aclanthology.org/2026.acl-long.773/)
- [논문 PDF](https://aclanthology.org/2026.acl-long.773.pdf)
- [DOI: 10.18653/v1/2026.acl-long.773](https://doi.org/10.18653/v1/2026.acl-long.773)
