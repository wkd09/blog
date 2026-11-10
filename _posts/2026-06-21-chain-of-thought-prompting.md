---
title: "Chain-of-Thought Prompting: LLM이 생각하는 과정을 쓰게 하면 왜 성능이 오를까?"
date: 2026-06-21 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - NLP
  - LLM
  - Prompting
  - Reasoning
source: "arXiv:2201.11903"
---

# Chain-of-Thought Prompting: LLM이 "생각하는 과정"을 쓰게 하면 왜 성능이 오를까?

이 글은 논문 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)를 바탕으로 정리한 글이다.

논문의 핵심은 간단하다. LLM에게 바로 답만 쓰게 하지 말고, 답에 도달하는 중간 추론 과정을 예시로 보여주면 복잡한 reasoning task 성능이 크게 좋아질 수 있다는 것이다. 논문은 이 방식을 `chain-of-thought prompting`, 줄여서 `CoT prompting`이라고 부른다.

![Chain-of-thought prompting 예시]({{ '/assets/images/blog/cot-fig1.png' | relative_url }})

*출처: 논문 Figure 1. Standard prompting은 답만 보여주지만, CoT prompting은 중간 추론 과정을 함께 보여준다.*

## 1. 이 논문이 나온 배경

LLM은 모델 크기를 키울수록 많은 NLP task에서 성능이 좋아졌다. 하지만 논문은 단순한 스케일업만으로는 산술 추론, 상식 추론, 기호 추론 같은 문제에서 충분한 성능을 내지 못했다고 설명한다. 특히 Introduction에서는 모델 크기 증가가 여러 장점을 주지만, challenging arithmetic, commonsense, symbolic reasoning에서는 높은 성능을 얻기에 충분하지 않았다고 정리한다. 출처는 논문 Section 1이다.

예를 들어 다음 문제를 보자.

```text
Roger has 5 tennis balls.
He buys 2 more cans of tennis balls.
Each can has 3 tennis balls.
How many tennis balls does he have now?
```

사람은 보통 이렇게 푼다.

```text
2 cans x 3 balls = 6 balls
5 + 6 = 11
answer = 11
```

그런데 기존 few-shot prompting은 보통 다음처럼 입력과 출력만 보여준다.

```text
Q: ...
A: The answer is 11.
```

이 방식은 단순 질의응답에서는 잘 작동할 수 있다. 하지만 여러 단계를 거쳐야 하는 문제에서는 모델이 어떤 중간 상태를 거쳐야 하는지 prompt 안에 충분히 드러나지 않는다. 논문은 기존 few-shot prompting이 reasoning ability가 필요한 task에서 잘 작동하지 않고, 모델 scale이 커져도 성능이 크게 개선되지 않는 경우가 있다고 설명한다. 출처는 논문 Section 1이다.

반대로 rationale을 달아 학습하거나 fine-tuning하는 방식도 있었다. 하지만 논문은 고품질 rationale 데이터를 대량으로 만드는 비용이 크다고 지적한다. 즉, 기존에는 두 선택지가 있었다.

- 답만 주는 few-shot prompting: 간단하지만 복잡한 reasoning에 약함
- rationale 기반 학습/fine-tuning: reasoning에는 도움될 수 있지만 데이터 작성 비용이 큼

CoT prompting은 이 둘 사이의 실용적인 절충안이다.

## 2. Chain-of-Thought Prompting이란?

Chain-of-Thought Prompting은 few-shot prompt의 예시를 `질문 -> 답` 형태가 아니라 `질문 -> 중간 추론 과정 -> 답` 형태로 구성하는 방법이다.

논문은 chain of thought를 "final output으로 이어지는 intermediate natural language reasoning steps"라고 정의한다. 출처는 논문 Section 1과 Section 2다.

기존 few-shot prompting은 이렇게 생겼다.

```text
Q: Roger has 5 tennis balls...
A: The answer is 11.

Q: The cafeteria had 23 apples...
A:
```

CoT prompting은 이렇게 바뀐다.

```text
Q: Roger has 5 tennis balls...
A: Roger started with 5 balls.
   2 cans of 3 tennis balls each is 6 tennis balls.
   5 + 6 = 11.
   The answer is 11.

Q: The cafeteria had 23 apples...
A:
```

여기서 중요한 점은 모델을 별도로 fine-tuning하지 않는다는 것이다. 논문은 off-the-shelf language model에 chain-of-thought 예시를 few-shot prompt로 넣었고, 이 논문 작성 과정에서 어떤 language model도 fine-tuning하지 않았다고 명시한다. 출처는 논문 Section 6 Discussion이다.

즉 CoT는 모델에 새로운 지식을 주입하는 방법이라기보다, 모델이 이미 가진 패턴과 능력을 prompt 형식으로 끌어내는 방법에 가깝다. 이 표현은 논문 결과를 바탕으로 한 해석이다.

## 3. 실험 결과 요약

논문은 크게 세 가지 reasoning 영역에서 CoT prompting을 평가한다.

- 산술 추론: GSM8K, SVAMP, ASDiv, AQuA, MAWPS
- 상식 추론: CSQA, StrategyQA, Date Understanding, Sports Understanding, SayCan
- 기호 추론: Last Letter Concatenation, Coin Flip

실험에는 UL2 20B, LaMDA, GPT-3, Codex, PaLM 계열 모델이 사용되었다. GPT-3는 text-ada-001, text-babbage-001, text-curie-001, text-davinci-002를 사용했고, 논문은 이들이 각각 350M, 1.3B, 6.7B, 175B 규모로 추정된다고 설명한다. PaLM은 8B, 62B, 540B를 사용했다. 출처는 논문 Section 3.1이다.

### 수학 추론

가장 유명한 결과는 GSM8K다. GSM8K는 초등학교 수준의 수학 word problem benchmark지만, 여러 단계를 거쳐야 해서 LLM에게 쉽지 않다.

논문 Table 1과 Table 2에 따르면 PaLM 540B는 GSM8K에서 standard prompting 17.9%에서 CoT prompting 56.9%로 상승했다. improvement는 +39.0%p다.

![GSM8K에서 PaLM 540B CoT 결과]({{ '/assets/images/blog/cot-fig2.png' | relative_url }})

*출처: 논문 Figure 2. PaLM 540B에서 CoT prompting이 GSM8K 성능을 크게 끌어올렸다.*

| 모델 | Benchmark | Standard prompting | CoT prompting | 출처 |
|---|---:|---:|---:|---|
| GPT-3 175B `text-davinci-002` | GSM8K | 15.6% | 46.9% | 논문 Table 2 |
| Codex `code-davinci-002` | GSM8K | 19.7% | 63.1% | 논문 Table 2 |
| PaLM 540B | GSM8K | 17.9% | 56.9% | 논문 Table 1, Table 2 |
| PaLM 540B | SVAMP | 69.4% | 79.0% | 논문 Table 2 |
| PaLM 540B | MAWPS | 79.2% | 93.3% | 논문 Table 2 |

논문은 이 결과를 세 가지 관점에서 해석한다. 출처는 논문 Section 3.2다.

- CoT prompting의 효과는 모델 scale에 의존한다.
- 더 복잡한 문제일수록 성능 향상이 크게 나타난다.
- GPT-3 175B와 PaLM 540B에서는 기존 supervised prior best와 비교할 만한 성능을 보였다.

다만 모든 수학 benchmark에서 동일하게 큰 폭으로 좋아진 것은 아니다. 예를 들어 PaLM 540B의 ASDiv는 72.1%에서 73.9%로 +1.8%p만 올랐고, AQuA는 25.2%에서 35.8%로 올랐다. 출처는 논문 Table 2다.

![수학 추론에서 scale에 따라 커지는 CoT 효과]({{ '/assets/images/blog/cot-fig4.png' | relative_url }})

*출처: 논문 Figure 4. CoT prompting의 이득은 작은 모델보다 큰 모델에서 뚜렷하게 나타난다.*

### 상식 추론

CoT는 수학 문제에만 적용되는 것이 아니다. 논문은 CSQA, StrategyQA, Date Understanding, Sports Understanding, SayCan에서도 실험했다.

PaLM 540B 기준 결과는 다음과 같다.

| Benchmark | Standard prompting | CoT prompting | 해석 | 출처 |
|---|---:|---:|---|---|
| CSQA | 78.1% | 79.9% | 상승폭은 작음 | 논문 Table 4 |
| StrategyQA | 68.6% | 77.8% | 명확한 향상 | 논문 Table 4 |
| Date Understanding | 49.0% | 65.3% | 날짜 계산/추론에서 향상 | 논문 Table 4 |
| Sports Understanding | 80.5% | 95.4% | 큰 향상 | 논문 Table 4 |
| SayCan | 80.8% | 91.7% | robot action planning 형태에서도 향상 | 논문 Table 4 |

![상식 추론에서의 CoT 결과]({{ '/assets/images/blog/cot-fig7.png' | relative_url }})

*출처: 논문 Figure 7. PaLM 기준으로 CoT prompting은 여러 commonsense reasoning benchmark에서 성능을 높였다.*

논문은 CoT의 언어 기반 특성 때문에 commonsense reasoning에도 적용 가능하다고 설명한다. 하지만 CSQA처럼 성능 향상이 매우 작았던 benchmark도 있다. 논문 Section 4 Results에서도 "gain was minimal on CSQA"라고 명시한다.

따라서 "CoT를 쓰면 모든 task가 크게 좋아진다"라고 말하면 과장이다. 더 정확한 정리는 이렇다.

- multi-step reasoning이 필요한 task에서는 도움이 될 가능성이 크다.
- 모델이 충분히 클수록 효과가 커지는 경향이 있다.
- 이미 standard prompting으로 잘 풀리는 task이거나, reasoning chain이 큰 도움이 되지 않는 task에서는 상승폭이 작을 수 있다.

### 기호 추론

논문은 두 가지 symbolic reasoning task도 실험했다.

첫 번째는 `Last Letter Concatenation`이다.

```text
Input: "Amy Brown"
Task: 각 단어의 마지막 글자를 이어 붙이기
Output: "yn"
```

두 번째는 `Coin Flip`이다.

```text
A coin is heads up.
Phoebe flips the coin.
Osvaldo does not flip the coin.
Is the coin still heads up?
```

여기서 중요한 점은 OOD length generalization이다. 예시는 2-step 또는 2-word 수준으로 보여주고, 테스트에서는 더 긴 3-step, 4-step 입력을 준다. 논문 Section 5는 Last Letter Concatenation에서 모델이 2-word 예시만 보고 3-word, 4-word 이름으로 일반화해야 한다고 설명한다.

PaLM 540B 결과는 다음과 같다.

| Task | Setting | Standard prompting | CoT prompting | 출처 |
|---|---|---:|---:|---|
| Last Letter Concatenation | 2 words, in-domain | 7.6% | 99.4% | 논문 Table 5 |
| Last Letter Concatenation | 3 words, OOD | 0.2% | 94.8% | 논문 Table 5 |
| Last Letter Concatenation | 4 words, OOD | 0.0% | 63.0% | 논문 Table 5 |
| Coin Flip | 2 flips, in-domain | 98.1% | 100.0% | 논문 Table 5 |
| Coin Flip | 3 flips, OOD | 49.3% | 98.6% | 논문 Table 5 |
| Coin Flip | 4 flips, OOD | 54.8% | 90.2% | 논문 Table 5 |

![기호 추론에서의 length generalization]({{ '/assets/images/blog/cot-fig8.png' | relative_url }})

*출처: 논문 Figure 8. CoT prompting은 더 긴 symbolic reasoning 입력으로 일반화하는 데 도움을 줬다.*

이 결과의 의미는 단순하다. 중간 과정을 명시하면 모델이 "정답 문자열을 바로 맞히는 문제"가 아니라 "단계별 조작을 따라가는 문제"로 task를 바꿔 풀 수 있다.

예를 들어 Last Letter Concatenation은 바로 답을 쓰게 하면 모델이 이름 전체를 보고 그럴듯한 문자열을 찍을 수 있다. 하지만 CoT 예시를 주면 모델은 다음 절차를 따라가도록 유도된다.

```text
The last letter of "Amy" is "y".
The last letter of "Brown" is "n".
Concatenating them is "yn".
The answer is yn.
```

논문은 OOD evaluation에서 standard prompting은 두 symbolic task 모두 실패했지만, CoT prompting에서는 충분히 큰 모델에서 upward scaling curve가 나타났다고 설명한다. 출처는 논문 Section 5 Results다.

## 4. 왜 CoT는 큰 모델에서만 잘 작동할까?

논문은 CoT 효과가 `scale-dependent`하다고 반복해서 말한다. 작은 모델에서는 CoT가 성능을 높이지 못하거나 오히려 낮추는 경우도 있었다.

수학 추론 Table 2를 보면 GPT 계열에서 작은 모델은 CoT가 오히려 나빠지는 경우가 있다.

| 모델 | GSM8K Standard | GSM8K CoT | 출처 |
|---|---:|---:|---|
| GPT 350M | 2.2% | 0.5% | 논문 Table 2 |
| GPT 1.3B | 2.4% | 0.5% | 논문 Table 2 |
| GPT 6.7B | 4.0% | 2.4% | 논문 Table 2 |
| GPT 175B | 15.6% | 46.9% | 논문 Table 2 |

논문 Section 3.2는 작은 모델들이 fluent하지만 illogical한 chain of thought를 생성해 standard prompting보다 낮은 성능을 냈다고 정성적으로 설명한다.

왜 그럴까?

CoT는 모델에 새로운 reasoning 알고리즘을 설치하지 않는다. prompt에 "이런 식으로 중간 과정을 써라"라는 형식을 보여줄 뿐이다. 따라서 모델 내부에 다음 능력이 어느 정도 있어야 한다.

- 자연어 지시와 예시 패턴을 이해하는 능력
- 중간 상태를 유지하는 능력
- 산술/상식/기호 조작을 단계적으로 수행하는 능력
- 마지막에 중간 추론과 일관된 답을 출력하는 능력

작은 모델은 chain 형식을 흉내 낼 수는 있지만, 각 단계가 논리적으로 맞는지 유지하기 어렵다. 반면 충분히 큰 모델은 prompt의 형식을 따라가면서 잠재된 reasoning 능력을 더 잘 드러낸다. 이 해석은 논문 Section 3.2와 Section 6의 scale-dependent 결과를 바탕으로 한 정리다.

## 5. 한계점

CoT는 강력하지만, 논문도 한계를 분명히 적고 있다. 출처는 논문 Section 6 Discussion이다.

### CoT가 실제 인간처럼 추론한다는 증거는 아니다

논문은 chain of thought가 인간의 사고 과정을 흉내 내지만, neural network가 실제로 reasoning하는지 여부는 답하지 않는다고 말한다. 즉, CoT 출력이 보인다고 해서 모델 내부에서 인간과 같은 방식의 추론이 일어난다고 결론 내릴 수는 없다.

### reasoning 과정이 항상 맞는 것은 아니다

CoT는 중간 과정을 생성하게 하지만, 그 과정이 항상 올바르다는 보장은 없다. 논문은 correct answer와 incorrect answer 모두에서 잘못된 reasoning path가 나타날 수 있다고 설명한다.

부록 Appendix D.2에서는 LaMDA 137B의 GSM8K 오답 50개를 분석한다. 논문은 오류 유형으로 calculator error, symbol mapping error, one step missing error 등을 제시한다. 예를 들어 Appendix D.2에 따르면 오답 chain 중 8%는 계산기 오류만 고치면 맞는 경우였고, 16%는 symbol mapping error, 22%는 one step missing error로 분류되었다.

### 그럴듯하지만 틀린 중간 추론이 나올 수 있다

CoT의 위험은 "말이 되는 것처럼 보이는 설명"을 만든다는 점이다. 모델은 자연어로 자연스럽게 설명을 이어갈 수 있지만, 그 설명이 실제로 답을 보장하지는 않는다. 논문 Section 6은 correct reasoning paths에 대한 보장이 없다고 명시한다.

### 큰 모델에서 주로 효과가 나타나므로 비용 문제가 있다

논문은 CoT reasoning이 큰 모델 scale에서만 emergent하게 나타나는 경향이 있어 실제 서비스에서 비용이 커질 수 있다고 말한다. 출처는 논문 Section 6이다.

추론 과정 자체도 출력 token을 늘린다. 논문이 이 비용을 별도의 token cost 실험으로 정량화하지는 않았지만, CoT가 중간 reasoning steps를 생성하는 방식이라는 점에서 inference output이 길어질 수 있다는 것은 방법론상 직접 따라오는 특성이다.

### 대규모 fine-tuning에는 rationale 데이터 비용이 크다

few-shot setting에서는 몇 개 예시에 chain을 붙이는 비용이 작다. 하지만 논문은 fine-tuning에 사용할 만큼 많은 rationale 데이터를 만들려면 annotation cost가 prohibitive할 수 있다고 지적한다. 출처는 논문 Section 6이다.

## 6. 이 논문의 핵심 의의

이 논문의 핵심 의의는 "모델을 크게 만들면 된다"가 아니다.

더 정확히는 "큰 모델이 가진 능력을 어떤 prompt 형식으로 끌어낼 것인가"를 보여준 논문이다.

표준 few-shot prompting은 모델 능력의 하한선만 보여줄 수 있다. 논문 Section 6은 standard prompting이 large language model capabilities의 lower bound만 제공한다고 표현한다. CoT는 같은 모델 checkpoint라도 prompt의 출력 형식을 바꾸면 훨씬 복잡한 task를 풀 수 있음을 보여줬다.

이후 reasoning prompting 연구는 이 아이디어를 여러 방향으로 확장했다.

- [Zero-shot CoT](https://arxiv.org/abs/2205.11916): few-shot 예시 없이 "Let's think step by step" 같은 문구로 step-by-step reasoning을 유도한다.
- [Self-Consistency](https://arxiv.org/abs/2203.11171): 여러 reasoning path를 샘플링하고 가장 일관된 답을 고른다.
- [ReAct](https://arxiv.org/abs/2210.03629): reasoning trace와 action을 번갈아 생성해 외부 도구나 환경과 상호작용한다.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601): 하나의 chain만 왼쪽에서 오른쪽으로 생성하지 않고, 여러 thought 후보를 탐색하고 평가한다.

위 후속 연구 연결은 원 논문의 실험 결과가 아니라, CoT 이후 등장한 연구 흐름을 별도 문헌 기준으로 정리한 것이다.

## 내가 이해한 핵심

Chain-of-Thought Prompting은 매우 단순하다.

```text
질문 -> 답
```

대신 이렇게 만든다.

```text
질문 -> 중간 추론 과정 -> 답
```

하지만 이 단순한 변경이 충분히 큰 모델에서는 큰 차이를 만든다. 논문 기준으로 PaLM 540B는 GSM8K에서 standard prompting 17.9%에서 CoT prompting 56.9%로 향상되었고, symbolic reasoning에서는 Last Letter Concatenation OOD 3-word task가 0.2%에서 94.8%로 상승했다. 출처는 각각 논문 Table 2와 Table 5다.

다만 CoT는 만능이 아니다. 작은 모델에서는 효과가 약하거나 역효과가 날 수 있고, 생성된 reasoning path가 항상 맞는 것도 아니다. 따라서 CoT를 사용할 때는 "모델이 진짜로 생각했다"라고 받아들이기보다, "중간 단계를 쓰도록 유도했을 때 답을 더 잘 찾는 경우가 있다" 정도로 이해하는 편이 안전하다.

## 한 줄 요약

Chain-of-Thought Prompting은 LLM에게 답만 요구하지 않고 중간 추론 과정을 쓰게 함으로써, 충분히 큰 모델 안에 잠재된 multi-step reasoning 능력을 끌어내는 prompting 방법이다.

## 참고 자료

- Jason Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903), NeurIPS 2022.
- Takeshi Kojima et al., [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), 2022.
- Xuezhi Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), 2022.
- Shunyu Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022.
- Shunyu Yao et al., [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601), 2023.
