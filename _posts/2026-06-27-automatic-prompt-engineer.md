---
title: "Automatic Prompt Engineer 논문 정리: 프롬프트 엔지니어링을 최적화 문제로 바꾸기"
date: 2026-06-27 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - Prompting
  - Agent
  - Paper
source: "arXiv:2211.01910"
---

# Large Language Models Are Human-Level Prompt Engineers

이 글은 논문 [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)를 바탕으로 정리한 글이다.

> Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, Jimmy Ba  
> ICLR 2023. [[Paper](https://arxiv.org/pdf/2211.01910)] [[Code](https://github.com/keirp/automatic_prompt_engineer)]

논문의 핵심 제안은 **Automatic Prompt Engineer**, 줄여서 `APE`다. 사람이 prompt를 감으로 고치고 몇 개만 실험하는 대신, LLM이 후보 instruction을 여러 개 만들고 실제 task 점수로 가장 좋은 instruction을 고른다.

핵심 흐름은 다음과 같다.

> APE는 `입출력 예시 -> 후보 instruction 생성 -> 후보별 성능 평가 -> 최고 점수 instruction 선택`을 자동화한 프롬프트 최적화 알고리즘이다.

중요한 점은 "프롬프트 자동 생성" 자체보다도, 논문이 프롬프트 엔지니어링을 **평가 가능한 black-box optimization 문제**로 바꿨다는 것이다.

![Automatic Prompt Engineer workflow]({{ '/assets/images/blog/ape-workflow.png' | relative_url }})

*출처: 논문 Figure 1. APE는 후보 instruction을 생성하고, target model에서 실행해 점수를 매긴 뒤, 가장 좋은 instruction을 선택한다.*

## 1. 이 논문이 다루는 문제

LLM 성능은 모델 자체뿐 아니라 프롬프트 품질에 크게 좌우된다. 같은 모델, 같은 데이터, 같은 task라도 instruction 표현이 조금 달라지면 결과가 크게 바뀔 수 있다.

예를 들어 antonym task에서 다음 instruction들은 사람에게는 거의 비슷해 보인다.

```text
write the antonym of the word.
write the opposite of the word given.
give the antonym of the word provided.
reverse the input.
```

하지만 모델 입장에서는 특정 표현이 훨씬 잘 작동할 수 있다. 사람은 자연어 의미가 비슷하면 비슷한 instruction이라고 느끼지만, LLM은 그 instruction을 같은 방식으로 실행하지 않을 수 있다.

논문은 이 문제를 이렇게 바라본다.

- LLM은 자연어 instruction을 실행하는 black-box computer다.
- 프롬프트는 이 black-box computer에 주는 자연어 프로그램이다.
- 좋은 프롬프트는 사람이 감으로 찾기보다, 후보를 만들고 실제 실행 점수로 고를 수 있다.

즉 APE의 출발점은 단순하다. 사람이 "이 프롬프트가 좋아 보인다"라고 판단하지 말고, **task에서 실제로 잘 되는 프롬프트를 자동으로 찾자**는 것이다.

## 2. 문제 정의

논문은 task를 입출력 예시 집합으로 본다.

$$
D_{train} = \{(Q, A)\}
$$

여기서 $Q$는 입력, $A$는 정답이다. 목표는 instruction $\rho$를 찾는 것이다.

$$
M([\rho; Q]) \approx A
$$

즉 모델 $M$에게 instruction $\rho$와 입력 $Q$를 이어 붙여 넣었을 때, 정답 $A$를 잘 출력하게 만드는 instruction을 찾는다.

논문은 이 목표를 다음 최적화 문제로 쓴다.

$$
\rho^* = \arg\max_{\rho} \mathbb{E}_{(Q,A)}[f(\rho, Q, A)]
$$

여기서 $f$는 점수 함수다. 예를 들면 exact match accuracy, log probability, TruthfulQA metric 같은 것이 들어갈 수 있다.

이 수식에서 중요한 부분은 $\rho$가 벡터나 hidden state가 아니라 **자연어 instruction**이라는 점이다. APE는 gradient로 연속 공간을 최적화하지 않고, LLM이 만든 자연어 후보들 위에서 탐색한다.

## 3. APE 알고리즘

APE는 크게 세 단계로 볼 수 있다.

```text
1. 입출력 예시를 보고 instruction 후보를 여러 개 생성한다.
2. 각 후보 instruction을 실제 target model에 넣어 task 점수를 계산한다.
3. 점수가 높은 후보를 선택하고, 필요하면 비슷한 후보를 다시 생성한다.
```

논문 Algorithm 1을 코드 느낌으로 단순화하면 다음과 같다.

```python
def ape(train_examples, generator_llm, target_llm, scorer, n_candidates=50):
    candidates = generator_llm.generate_instructions(
        examples=train_examples,
        n=n_candidates,
    )

    scored = []
    for instruction in candidates:
        score = scorer(
            model=target_llm,
            instruction=instruction,
            examples=train_examples,
        )
        scored.append((instruction, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0]
```

실제 논문에서는 여기에 adaptive filtering과 iterative resampling이 붙는다. 모든 후보를 모든 example에 대해 끝까지 평가하면 비용이 크기 때문에, 처음에는 작은 subset으로 빠르게 걸러내고, 가능성이 있는 후보에 더 많은 평가 비용을 쓰는 식이다.

## 4. 후보 instruction 생성

APE의 첫 단계는 LLM에게 instruction 후보를 만들게 하는 것이다.

예를 들어 다음 입출력 예시가 있다고 하자.

```text
Input: prove
Output: disprove

Input: on
Output: off

The instruction was:
```

그러면 LLM은 다음 같은 후보를 생성할 수 있다.

```text
write the antonym of the word.
give the opposite of the input word.
write the word with the opposite meaning.
```

논문은 candidate proposal 방식을 크게 두 가지로 나눈다.

첫 번째는 **forward generation**이다. 입출력 예시를 먼저 보여준 뒤, "instruction은 무엇이었을까?"를 뒤에서 완성하게 한다. 대부분의 left-to-right language model에서 바로 쓸 수 있다.

두 번째는 **reverse generation**이다. instruction이 들어갈 자리를 중간에 비워두고, infilling 가능한 모델이 빈칸을 채우게 한다. 논문은 T5, GLM, InsertGPT 같은 infilling model을 예로 든다.

이 단계의 핵심은 LLM을 정답 생성기가 아니라 **proposal distribution**으로 쓴다는 것이다. LLM이 완벽한 instruction 하나를 맞혀야 하는 게 아니다. 그럴듯한 후보를 충분히 많이 만들고, 다음 단계에서 실제 점수로 고르면 된다.

## 5. 후보 instruction 평가

후보를 만들었으면, 이제 각 instruction을 실제 target model에 붙여서 실행한다.

논문에서 주로 다루는 점수 함수는 두 가지다.

첫 번째는 **execution accuracy**다.

$$
f_{exec}(\rho, Q, A) = \mathbf{1}[M([\rho; Q]) = A]
$$

instruction $\rho$를 붙였을 때 모델 출력이 정답 $A$와 같으면 1점, 아니면 0점이다. 단순하지만 task success를 직접 본다는 장점이 있다.

두 번째는 **log probability**다.

$$
f_{logprob}(\rho, Q, A) = \log P(A \mid [\rho; Q])
$$

모델이 정답 문자열에 얼마나 높은 확률을 주는지 평가한다. 출력이 exact match로 맞지 않아도 더 부드러운 신호를 줄 수 있다.

하지만 논문은 Instruction Induction 실험에서 execution accuracy가 test performance와 더 잘 맞았다고 보고한다. 이 부분은 실무적으로도 중요하다. agent나 tool-calling prompt를 최적화할 때도 "정답에 높은 확률을 줬는가"보다 **실제로 원하는 행동을 했는가**가 더 직접적인 metric인 경우가 많다.

## 6. Iterative Monte Carlo Search

APE는 선택적으로 상위 후보 주변을 다시 탐색한다.

과정은 이렇다.

```text
1. 후보 instruction을 평가한다.
2. 상위 k% 후보만 남긴다.
3. LLM에게 상위 후보와 의미가 비슷한 변형을 다시 만들게 한다.
4. 새 후보를 다시 평가한다.
```

논문은 이 방식을 `Iterative Monte Carlo Search`라고 부른다. 좋은 후보 근처를 local search하는 셈이다.

![APE sample size and iterative search]({{ '/assets/images/blog/ape-iterative-search.png' | relative_url }})

*출처: 논문 Figure 6 일부. Iterative search는 후보 집합의 평균 품질을 올릴 수 있지만, 최고 성능 후보는 초기 후보에서 이미 발견되는 경우도 많다.*

재밌는 점은 iterative search가 항상 큰 이득을 주지는 않았다는 것이다. 논문은 iterative generation이 proposal set의 전반적인 품질은 올리지만, 가장 높은 점수의 instruction은 여러 stage를 거쳐도 그대로 남는 경향이 있다고 분석한다. 그래서 기본 APE는 복잡한 반복 탐색보다, 먼저 좋은 후보를 충분히 뽑고 평가하는 쪽에 가깝다.

## 7. 실험 1: Instruction Induction

Instruction Induction은 입출력 예시를 보고 규칙을 추론하는 task다. 예를 들면 antonym, pluralization, translation, sentiment, first letter, second letter 같은 문제가 있다.

논문은 24개 Instruction Induction task에서 APE를 평가했다. 결과적으로 APE는 24/24 task에서 human-level 또는 그 이상의 zero-shot performance를 보였다. Figure 1에 따르면 24개 task의 interquartile mean 기준으로 APE with InstructGPT는 0.81, human prompt engineer는 0.75 수준이다.

![APE instruction induction results]({{ '/assets/images/blog/ape-instruction-induction.png' | relative_url }})

*출처: 논문 Appendix figure. 24개 Instruction Induction task에서 APE는 많은 task에서 human prompt와 비슷하거나 더 높은 성능을 낸다.*

여기서 중요한 비교는 greedy generation과 APE의 차이다. LLM이 instruction 하나를 바로 생성하게 하는 것보다, 여러 후보를 만들고 평가해서 고르는 search/selection 과정이 성능을 크게 끌어올렸다.

즉 성능 향상의 핵심은 "LLM이 좋은 문장을 한 번에 잘 쓴다"가 아니다. **LLM이 그럴듯한 후보를 많이 만들고, 평가 루프가 그중 좋은 것을 고른다**는 데 있다.

## 8. 실험 2: Few-shot In-context Learning

APE로 찾은 instruction은 zero-shot에만 쓰이지 않는다. 논문은 APE instruction을 standard few-shot prompt 앞에 붙이는 실험도 했다.

결과는 24개 중 21개 task에서 standard in-context learning과 비슷하거나 더 좋은 성능이었다. 다만 Rhymes, Large Animal, Second Letters 같은 일부 task에서는 오히려 성능이 떨어졌다.

이 결과는 APE의 중요한 한계를 보여준다.

zero-shot setting에서 고른 instruction이 few-shot setting에서도 항상 최적이라는 보장은 없다. prompt는 주변 context와 상호작용한다. 따라서 APE를 쓸 때는 "어떤 setting에서 평가했는가"가 중요하다.

실무적으로 말하면 system prompt만 따로 최적화했을 때는 좋아 보였는데, 실제 RAG context나 tool schema와 붙이면 나빠질 수 있다. 그러면 평가 데이터와 scorer를 실제 배포 prompt 구조에 맞춰 다시 설계해야 한다.

## 9. 실험 3: BIG-Bench

논문은 더 어려운 task에서도 APE를 평가하기 위해 BIG-Bench Instruction Induction, 즉 `BBII`라는 21개 task subset을 구성했다.

여기에는 다음 성격의 task들이 포함된다.

- emotional understanding
- reading comprehension
- summarization
- arithmetic reasoning
- commonsense reasoning
- symbolic and logical reasoning

결과적으로 APE는 21개 중 17개 task에서 human-written default prompt와 비슷하거나 더 좋은 성능을 보였다.

Instruction Induction 24/24보다 낮은 수치인데, 이는 BIG-Bench 쪽 task가 더 복잡하고 reasoning 성격이 강하기 때문으로 볼 수 있다. APE가 좋은 instruction을 찾을 수는 있지만, instruction만으로 모델의 reasoning 한계를 완전히 넘지는 못한다.

## 10. 실험 4: Zero-shot Chain-of-Thought

이 논문에서 특히 흥미로운 부분은 APE가 task instruction 전체뿐 아니라, prompt의 짧은 phrase도 최적화했다는 점이다.

기존 Zero-shot-CoT의 대표 문구는 다음이다.

```text
Let's think step by step.
```

논문은 APE로 answer-prefix를 탐색해서 다음 문구를 찾았다.

```text
Let's work this out in a step by step way to be sure we have the right answer.
```

이 문구는 MultiArith에서 78.7에서 82.0으로, GSM8K에서 40.7에서 43.0으로 성능을 올렸다고 보고된다.

여기서 얻을 수 있는 교훈은 크다. APE는 system prompt 전체를 갈아엎는 도구가 아니라, prompt의 특정 부분만 최적화하는 데에도 쓸 수 있다. 예를 들어 agent에서는 다음 요소들을 따로 최적화할 수 있다.

- tool 선택 전에 붙이는 짧은 policy phrase
- JSON 출력 전 format reminder
- code repair 전 patch instruction
- judge prompt의 scoring rubric
- router prompt의 fallback instruction

즉 APE는 "프롬프트 전체 자동 작성기"라기보다, **프롬프트 구성 요소를 평가 루프 안에서 개선하는 방법**에 가깝다.

## 11. 실험 5: TruthfulQA

TruthfulQA에서는 APE를 정답률만 높이는 용도가 아니라, 모델의 답변 스타일을 조정하는 데 사용했다.

논문은 세 가지 metric을 최적화했다.

- `% True`: 답변이 진실한가
- `% Info`: 답변이 정보를 제공하는가
- `% True + % Info`: 진실성과 정보성의 조합

APE는 200개 후보만으로 human-engineered "help" prompt보다 좋은 성능을 보였고, training set에서 고른 상위 instruction들이 test set에도 잘 일반화되었다고 보고한다.

하지만 여기에는 중요한 함정이 있다. truthfulness만 강하게 최적화하면 모델이 "No comment"류의 회피 답변을 하도록 유도될 수 있다. 그러면 거짓말은 줄지만 정보성도 같이 떨어진다. 논문도 truthfulness와 informativeness 사이의 trade-off를 명확히 다룬다.

이 사례는 APE의 가장 중요한 위험을 보여준다.

> APE는 좋은 답변을 찾는 것이 아니라, 우리가 정의한 score function을 최대화하는 instruction을 찾는다.

metric이 부정확하면 APE는 좋은 프롬프트가 아니라 metric을 해킹하는 프롬프트를 고를 수 있다.

## 12. 후보 수를 늘리면 얼마나 좋아질까?

논문은 posterior sample size, 즉 후보 instruction 수를 늘렸을 때 성능이 어떻게 변하는지도 분석했다.

![APE posterior sample size]({{ '/assets/images/blog/ape-sample-size.png' | relative_url }})

*출처: 논문 Figure 7. 후보 수를 늘리면 성능은 올라가지만, 증가폭은 점점 줄어든다.*

후보 수를 4개에서 128개까지 늘리면 성능은 대체로 좋아진다. 하지만 수익은 점점 줄어든다. 논문은 64개 샘플에서 human-level에 도달했고, 기본 실험에서는 50개 후보를 사용했다고 설명한다.

실무적으로는 처음부터 후보 500개를 만들기보다 다음 흐름이 현실적이다.

```text
1. 후보 30~50개 생성
2. 작은 validation set으로 빠른 평가
3. top-k 후보만 큰 validation set에서 재평가
4. 비용, latency, format error까지 포함해 최종 선택
```

APE의 비용은 후보 수와 평가 데이터 수에 거의 비례해서 늘어난다. 그래서 후보를 많이 뽑는 것보다, 좋은 scorer와 효율적인 filtering이 더 중요해진다.

## 13. Agent prompt 최적화에 적용하기

이 논문은 지금의 LLMOps나 AgentOps 관점에서도 바로 연결된다.

agent에서 프롬프트는 단순 문장이 아니다. tool 선택, 출력 형식, 실패 복구, 위험 회피, 비용 제어 같은 정책을 담은 자연어 프로그램이다.

예를 들어 tool-calling agent에서 APE를 쓴다면 최적화 대상은 다음처럼 나눌 수 있다.

- system prompt
- tool selection prompt
- code repair prompt
- judge prompt
- reflection prompt
- router prompt

평가 데이터는 다음처럼 만들 수 있다.

| 입력 | 기대 결과 |
|---|---|
| 사용자 요청 | 호출해야 할 tool |
| 취약한 코드 | 수정된 코드와 통과해야 할 테스트 |
| 애매한 질문 | clarification 여부 |
| 외부 API 요청 | 올바른 JSON schema |
| 실패 로그 | 적절한 retry 또는 fallback |

scorer는 task에 맞게 조합할 수 있다.

$$
score =
0.35 \cdot task\_success
+ 0.25 \cdot tool\_selection\_accuracy
+ 0.20 \cdot json\_valid\_rate
+ 0.10 \cdot latency\_score
+ 0.10 \cdot cost\_score
$$

위 식은 논문에 있는 식이 아니라, agent prompt 최적화에 맞춘 예시다. 논문 자체는 task에 따라 execution accuracy, log probability, TruthfulQA metric 등을 사용했다.

핵심은 프롬프트를 버전 관리 가능한 artifact로 보고, 변경할 때마다 자동 평가를 통과하게 만드는 것이다.

```text
prompt candidate
  -> eval dataset
  -> task metric
  -> cost/latency penalty
  -> selected prompt
  -> versioned release
```

이렇게 보면 APE는 prompt engineering을 "글쓰기"에서 "실험 가능한 optimization loop"로 옮긴다.

## 14. 한계

첫 번째 한계는 **metric dependency**다. APE는 score function을 최대화한다. score가 잘못 설계되면 좋은 프롬프트가 아니라 score를 잘 속이는 프롬프트가 선택될 수 있다. TruthfulQA에서 회피 답변이 truthfulness를 올릴 수 있는 사례가 대표적이다.

두 번째 한계는 **overfitting**이다. train examples에서 잘 되는 instruction이 실제 배포 입력에서도 항상 잘 되는 것은 아니다. 특히 few-shot setting처럼 prompt context가 달라지면 성능이 바뀔 수 있다.

세 번째 한계는 **비용**이다. 후보를 여러 개 만들고, 각 후보를 여러 example에 대해 평가해야 한다. target model이 비싸거나 latency가 크면 APE 자체가 부담이 된다. adaptive filtering이 필요한 이유도 여기에 있다.

네 번째 한계는 **모델 시대성**이다. 논문은 GPT-3/InstructGPT 계열 API 중심으로 실험했다. APE라는 관점은 지금도 유효하지만, 논문 속 절대 성능 수치를 최신 모델에 그대로 일반화하면 안 된다.

다섯 번째 한계는 **instruction만으로 해결할 수 없는 task**다. 모델이 필요한 지식이나 reasoning 능력을 갖고 있지 않다면, prompt search만으로는 한계가 있다. 이 경우 RAG, tool use, fine-tuning, data improvement가 함께 필요하다.

## 15. 내가 이해한 핵심

APE를 처음 보면 LLM에게 prompt 작성을 맡기는 방법처럼 보인다. 하지만 내가 이해한 핵심은 prompt engineering을 다음과 같은 optimization loop로 바꿨다는 점이다.

```text
기존 방식:
사람이 감으로 prompt 작성
-> 몇 개 테스트
-> 마음에 들면 사용

APE 방식:
입출력 예시 준비
-> 후보 prompt 대량 생성
-> metric 기반 평가
-> 가장 좋은 prompt 선택
-> 필요하면 반복 개선
```

즉 APE의 메시지는 이렇다.

> 프롬프트는 감으로 관리하지 말고, 후보 생성, 자동 평가, 선택, 버전 관리의 대상으로 다뤄야 한다.

지금 보면 APE는 LLM-as-optimizer, prompt search, agent prompt evaluation loop의 초기 형태에 가깝다. 특히 tool-calling agent, RAG router, code repair agent, evaluator prompt를 만들 때 좋은 baseline으로 삼을 수 있다.

프롬프트가 자연어라서 느슨해 보일 뿐, 실제로는 모델 행동을 결정하는 프로그램이다. 그렇다면 좋은 프로그램을 테스트 없이 배포하지 않듯이, 좋은 프롬프트도 평가 없이 배포하지 않는 게 맞다.
