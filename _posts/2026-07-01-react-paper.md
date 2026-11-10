---
title: "ReAct 논문 정리: LLM Agent는 왜 생각하면서 행동해야 할까?"
date: 2026-07-01 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - NLP
  - LLM
  - Agent
  - Prompting
  - Reasoning
source: "arXiv:2210.03629"
---

# ReAct: LLM Agent는 왜 생각하면서 행동해야 할까?

이 글은 논문 [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)를 바탕으로 정리한 글이다.

> Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao  
> ICLR 2023. [[Paper](https://arxiv.org/pdf/2210.03629)] [[Project](https://react-lm.github.io/)]

핵심 관심사는 하나다.

> LLM은 생각만 잘해서도 부족하고, 행동만 잘해서도 부족하다. 생각하면서 행동하고, 행동 결과를 다시 보고 생각을 고쳐야 한다.

기존 연구는 크게 두 흐름으로 나뉘어 있었다.

- `Reasoning`: Chain-of-Thought처럼 중간 추론 과정을 생성해서 답을 찾는다.
- `Acting`: 검색, 웹 탐색, 게임 내 이동, API 호출 같은 action을 생성해서 외부 환경과 상호작용한다.

ReAct는 이 둘을 분리하지 않는다. LLM이 `Thought`, `Action`, `Observation`을 번갈아 생성하게 만들어, reasoning이 action을 안내하고 action 결과가 다시 reasoning을 보정하게 한다.

![ReAct Figure 1 overview]({{ '/assets/images/blog/react-fig1-overview.png' | relative_url }})

*출처: 논문 Figure 1. ReAct는 reasoning trace와 action을 번갈아 생성하면서 HotpotQA와 ALFWorld에서 더 안정적으로 문제를 해결한다.*

## 1. CoT만으로는 왜 부족한가?

Chain-of-Thought는 다음 구조로 동작한다.

```text
Question
-> Thought
-> Thought
-> Thought
-> Answer
```

겉보기에는 좋아 보인다. 모델이 바로 답하지 않고 중간 사고 과정을 쓰기 때문이다. 하지만 CoT에는 중요한 약점이 있다.

> 모델이 틀린 사실을 바탕으로 생각하면, 그 뒤 reasoning 전체가 같이 무너진다.

논문 Figure 1의 HotpotQA 예시를 보자.

```text
Question:
Aside from the Apple Remote, what other device can control the program
Apple Remote was originally designed to interact with?
```

CoT는 내부 지식만으로 다음처럼 추론한다.

```text
Apple Remote는 Apple TV와 관련 있다.
Apple TV는 iPhone, iPad, iPod Touch로 제어할 수 있다.
따라서 답은 iPhone, iPad, iPod Touch다.
```

문제는 첫 전제가 틀렸다는 것이다. Apple Remote가 원래 제어하려고 설계된 대상은 Apple TV가 아니라 `Front Row`였다. 한 번 틀린 사실이 reasoning chain 앞쪽에 들어가면, 뒤쪽 추론은 아무리 그럴듯해도 오답으로 간다.

논문은 이를 `hallucination`과 `error propagation` 문제로 본다.

- CoT는 생각을 잘한다.
- 하지만 외부 세계로 확인하지 않는다.
- 그래서 틀린 사실도 자신 있게 이어간다.

## 2. Action만으로는 왜 부족한가?

반대로 Act-only는 외부 도구를 사용한다.

```text
Question
-> Action: Search[Apple Remote]
-> Observation
-> Action: Search[Front Row]
-> Observation
-> Action: Finish[answer]
```

외부 정보를 가져오므로 CoT보다 사실성은 좋아질 수 있다. 하지만 Act-only에는 중간 reasoning이 없다. 즉, 검색 결과가 애매할 때 다음 검색어를 어떻게 바꿀지, 여러 observation 중 무엇이 중요한지, 언제 답을 끝내야 하는지를 명시적으로 정리하지 못한다.

논문 Figure 1에서도 Act-only는 검색을 수행하지만 최종 답을 잘못 낸다. 정보는 가져오지만, 그 정보를 해석하고 다음 계획을 수정하는 능력이 약하기 때문이다.

정리하면 이렇다.

| 방식 | 장점 | 한계 |
|---|---|---|
| Standard Prompting | 간단하다 | 복잡한 추론에 약하다 |
| CoT | reasoning trace를 만든다 | 외부 정보 확인이 없어 hallucination에 취약하다 |
| Act-only | 외부 정보에 접근한다 | 계획과 해석이 약하다 |
| ReAct | reasoning과 acting을 결합한다 | prompt 길이와 반복 loop 문제가 생길 수 있다 |

## 3. ReAct의 기본 구조

ReAct는 CoT와 Act-only를 합친다.

```text
Question
-> Thought
-> Action
-> Observation
-> Thought
-> Action
-> Observation
-> Finish
```

여기서 세 요소의 역할은 명확하다.

| ReAct 요소 | 의미 | 현대 Agent 구현 |
|---|---|---|
| `Thought` | 내부 추론, 계획, 상태 정리 | LLM reasoning / planner |
| `Action` | 외부 환경에 실행하는 요청 | tool call |
| `Observation` | action이 돌려준 결과 | tool result |
| `Finish` | 최종 답변 제출 | final answer |

예를 들어 Figure 1의 HotpotQA 문제는 다음처럼 진행된다.

```text
Thought 1:
Apple Remote가 원래 어떤 프로그램과 상호작용하려 했는지 찾아야 한다.

Action 1:
Search[Apple Remote]

Observation 1:
Apple Remote는 Front Row media center program을 제어하기 위해 만들어졌다.

Thought 2:
이제 Front Row가 어떤 장치로 제어 가능한지 찾아야 한다.

Action 2:
Search[Front Row]

Observation 2:
검색 결과가 애매하고 Front Row (software)가 비슷한 항목으로 나온다.

Thought 3:
Front Row가 안 나왔으니 Front Row (software)를 검색해야 한다.

Action 3:
Search[Front Row (software)]

Observation 3:
Front Row는 미디어 센터 소프트웨어다.

Thought 4:
Front Row는 Apple Remote나 keyboard function keys로 제어된다.

Finish:
keyboard function keys
```

핵심은 observation을 본 뒤 thought가 바뀐다는 점이다. ReAct는 처음부터 완벽한 계획을 세우는 방식이 아니라, 행동 결과를 받아 가면서 계획을 수정하는 방식이다.

## 4. Thought, Action, Observation의 차이

ReAct에서 가장 중요한 구분은 `Thought`와 `Action`을 섞지 않는 것이다.

`Thought`는 외부 세계를 바꾸지 않는다.

```text
Thought:
먼저 Apple Remote가 원래 어떤 프로그램을 제어하려고 했는지 찾아야 한다.
```

이건 검색이 아니다. 모델 내부의 계획이다. Thought는 문제를 작은 단계로 나누고, 다음 action을 계획하고, observation에서 중요한 정보를 뽑고, 현재 진행 상황을 추적하고, 실패했을 때 계획을 수정한다.

반대로 `Action`은 외부 환경에 실제로 요청하는 것이다. 논문에서 Wikipedia API에 대해 사용한 action space는 세 가지다.

| Action | 역할 |
|---|---|
| `Search[entity]` | 특정 Wikipedia 문서를 검색한다 |
| `Lookup[string]` | 현재 문서 안에서 특정 문자열을 찾는다 |
| `Finish[answer]` | 최종 답을 제출한다 |

`Observation`은 action의 결과다.

```text
Action:
Search[Apple Remote]

Observation:
Apple Remote는 Front Row를 제어하기 위해 만들어졌다.
```

중요한 점은 observation은 LLM이 상상한 문장이 아니라 환경이 돌려준 결과라는 것이다. 그래서 observation은 ReAct에서 grounding 역할을 한다.

잘못된 구조는 이런 것이다.

```text
Action:
Search[Apple Remote가 원래 무엇을 제어했는지 생각해보자]
```

검색 도구에는 검색어만 들어가야 한다. 왜 검색해야 하는지는 Thought에 있고, 실제 실행 가능한 명령은 Action에 있어야 한다.

## 5. Observation이 핵심이다

ReAct를 단순히 `Thought -> Action -> Observation`이라고 외우면 반만 이해한 것이다. 진짜 핵심은 observation이 다음 thought를 바꾼다는 데 있다.

CoT는 다음 흐름이다.

```text
Question
-> Thought 1
-> Thought 2
-> Thought 3
-> Answer
```

여기서는 한 번도 "이 생각이 맞는가?"를 확인하지 않는다.

ReAct는 다르다.

```text
Thought
-> Action
-> Observation
-> Thought 수정
-> Action
-> Observation
-> Thought 수정
```

예를 들어 ALFWorld에서 목표가 "pepper shaker를 drawer 위에 놓기"라고 하자.

```text
Thought:
pepper shaker는 cabinet이나 countertop에 있을 가능성이 높다.

Action:
Go to cabinet 1

Observation:
cabinet 1에는 vase가 있다.

Thought:
cabinet에는 없다. countertop을 찾아야 한다.
```

Observation은 LLM에게 "방금 가정이 틀렸을 수 있다"는 피드백을 준다. 그래서 ReAct의 강점은 action 자체가 아니라, action 결과를 받아 reasoning을 업데이트하는 피드백 루프에 있다.

현대 Agent 프레임워크로 바꾸면 다음과 같다.

| ReAct | LangGraph | OpenAI Agents SDK | MCP |
|---|---|---|---|
| `Action` | Node / tool call | Tool call | Tool call |
| `Observation` | Node output | Tool result | Tool response |
| `Thought` | 다음 node 선택 / reasoning | 모델의 다음 판단 | 다음 tool 선택 |

즉, 오늘날 tool-calling agent의 기본 루프는 ReAct와 거의 같은 구조를 가진다.

## 6. Prompt 설계 관점

ReAct는 모델 구조를 바꾸는 논문이 아니다. 핵심은 prompt로 LLM이 `Thought`, `Action`, `Observation` 패턴을 따르게 만드는 것이다.

기본 prompt 구조는 다음과 같다.

```text
Question: ...
Thought 1: ...
Action 1: ...
Observation 1: ...
Thought 2: ...
Action 2: ...
Observation 2: ...
...
Thought N: ...
Action N: Finish[answer]
```

논문은 HotpotQA에는 6개, FEVER에는 3개의 ReAct trajectory 예시를 few-shot prompt로 넣었다. 예시는 단순한 설명이 아니라 실제 trajectory다. 모델은 이를 보고 다음 패턴을 배운다.

- 먼저 생각한다.
- 실행 가능한 action을 고른다.
- observation을 읽는다.
- observation에 따라 계획을 수정한다.
- 충분하면 `Finish`로 종료한다.

예를 들어 지하철 데이터 분석 agent라면 ReAct prompt는 이런 식으로 만들 수 있다.

```text
You are a subway data analysis agent.

Available actions:
SearchData[line, station, date]
Calculate[value]
Finish[answer]

Question:
2024년 2호선에서 승차 승객이 가장 많은 역은?

Thought 1:
2호선의 역별 승차총승객수를 조회해야 한다.

Action 1:
SearchData[line="2호선", date="2024"]

Observation 1:
강남역 123000명, 잠실역 98000명, 홍대입구역 87000명 ...

Thought 2:
승차총승객수가 가장 큰 역은 강남역이다.

Action 2:
Finish[강남역]
```

구현할 때는 다음 규칙이 중요하다.

1. 사용 가능한 action을 제한한다.
2. action 출력 형식을 엄격하게 정한다.
3. observation을 context에 반드시 추가한다.
4. 같은 action 반복을 감지한다.
5. 최대 step 수를 둔다.
6. `Finish`가 나오면 종료한다.

요즘 OpenAI API나 LangGraph에서는 ReAct prompt를 문자열로 직접 만들지 않아도 된다. tool schema와 tool result 메시지가 이 구조를 대신한다. 그래도 개념적으로는 여전히 ReAct다.

```text
LLM
-> Tool Call
-> Tool Result
-> LLM
-> Tool Call
-> Tool Result
-> Final Answer
```

## 7. HotpotQA와 FEVER 실험 결과

논문은 knowledge-intensive reasoning task로 HotpotQA와 FEVER를 사용했다.

![ReAct HotpotQA FEVER results]({{ '/assets/images/blog/react-table-hotpot-fever.png' | relative_url }})

*출처: 논문 Table 1. ReAct 단독은 항상 최고는 아니지만, Act-only보다 안정적이고 CoT-SC와 결합할 때 강해진다.*

주요 결과는 다음과 같다.

| Method | HotpotQA EM | FEVER Acc |
|---|---:|---:|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC -> ReAct | 34.2 | 64.6 |
| ReAct -> CoT-SC | 35.1 | 62.0 |

여기서 중요한 해석은 세 가지다.

첫째, ReAct는 Act-only보다 두 task 모두에서 좋았다. reasoning이 action을 안내하기 때문이다.

둘째, ReAct 단독이 항상 CoT보다 좋은 것은 아니다. HotpotQA에서는 CoT보다 낮고 FEVER에서는 높다. ReAct는 외부 정보를 확인하므로 hallucination은 줄이지만, 검색 실패나 반복 행동 같은 새로운 오류에 취약하다.

셋째, ReAct와 CoT-SC를 결합하면 가장 강해진다. CoT-SC는 내부 reasoning 다양성을 활용하고, ReAct는 외부 정보로 grounding한다. 논문은 두 방식의 실패 양상이 다르기 때문에 결합 효과가 난다고 본다.

![CoT-SC and ReAct combination]({{ '/assets/images/blog/react-cotsc-trials.png' | relative_url }})

*출처: 논문 Figure 3. CoT-SC와 ReAct를 결합하면 단일 방법보다 더 높은 성능을 낼 수 있다.*

## 8. ReAct가 hallucination을 줄이는 이유

논문 Table 2의 failure analysis는 ReAct의 장단점을 잘 보여준다.

![ReAct failure analysis]({{ '/assets/images/blog/react-failure-analysis.png' | relative_url }})

*출처: 논문 Table 2. CoT는 hallucination 실패가 많고, ReAct는 search result error와 reasoning error가 주요 실패 원인이다.*

성공 사례 중에서도 ReAct는 더 정확한 reasoning trace와 사실을 포함하는 비율이 높았다. 실패 사례를 보면 차이가 더 뚜렷하다.

| Failure type | ReAct | CoT |
|---|---:|---:|
| Reasoning error | 47% | 16% |
| Search result error | 23% | - |
| Hallucination | 0% | 56% |
| Label ambiguity | 29% | 28% |

CoT는 내부 기억에 의존한다. 그래서 그럴듯하지만 틀린 사실을 만들어낼 수 있다. 반면 ReAct는 Wikipedia 같은 외부 knowledge base의 observation을 읽는다. 이 때문에 fact hallucination은 줄어든다.

하지만 ReAct도 완벽하지 않다. observation이 비어 있거나 검색어가 잘못되면 reasoning이 흔들린다. 논문에서 ReAct 실패 중 `Search result error`가 23%였다는 점이 중요하다.

즉 ReAct의 성능은 tool 품질에 의존한다.

```text
좋은 Observation -> 좋은 Reasoning
나쁜 Observation -> 나쁜 Reasoning
```

그래서 현대 agent에서는 retriever, reranker, tool result validator가 중요해진다.

## 9. 최신 정보가 필요한 질문에서의 ReAct

ReAct가 유용한 대표 상황은 모델 내부 지식이 오래되었을 때다.

![ReAct updated knowledge example]({{ '/assets/images/blog/react-hotpot-updated.png' | relative_url }})

*출처: 논문 Figure 4. ReAct는 검색을 통해 outdated label과 다른 최신 정보를 반영할 수 있다.*

Figure 4의 예시는 Cirque du Soleil show `Mystere`가 열리는 호텔의 객실 수를 묻는다. 기존 HotpotQA label은 오래된 정보를 기준으로 되어 있고, ReAct는 검색을 통해 더 최신의 객실 수 정보를 찾아낸다.

이 예시는 ReAct의 중요한 성질을 보여준다.

- 모델 내부 기억은 오래될 수 있다.
- 데이터셋 label도 오래될 수 있다.
- tool observation은 현재 외부 세계를 반영할 수 있다.

물론 observation이 항상 정답이라는 뜻은 아니다. 검색 결과도 틀릴 수 있고, 오래될 수 있고, 질문과 무관할 수 있다. 그래서 좋은 agent는 observation을 그대로 믿는 것이 아니라, observation의 출처와 품질을 평가해야 한다.

## 10. ALFWorld에서의 ReAct

ReAct는 지식 검색뿐 아니라 의사결정 task에서도 평가되었다. ALFWorld는 텍스트 기반 가정 환경에서 목표를 수행하는 benchmark다.

예를 들어 목표가 다음과 같다고 하자.

```text
Put some peppershaker on a drawer.
```

Act-only는 행동만 반복하다가 물건이 없는 곳에서 계속 집으려 할 수 있다. 반면 ReAct는 먼저 탐색 전략을 세운다.

```text
Thought:
pepper shaker는 cabinet, countertop, shelf 같은 곳에 있을 가능성이 높다.

Action:
Go to cabinet 1

Observation:
cabinet 1에는 vase가 있다.

Thought:
countertop을 찾아보자.
```

여기서 Thought는 단순 설명이 아니라 탐색 정책이다. 논문은 decision making task에서 thought가 목표 분해, 진행 상황 추적, 다음 subgoal 결정, 상식 기반 추론에 쓰인다고 설명한다.

![ReAct ALFWorld human thought edit]({{ '/assets/images/blog/react-alfworld-edit.png' | relative_url }})

*출처: 논문 Figure 5. ALFWorld에서는 thought가 탐색 전략, 상태 추적, 다음 subgoal 결정에 직접 영향을 준다.*

ALFWorld 결과에서도 ReAct는 전체 성공률에서 Act-only와 BUTLER baseline보다 좋은 성능을 보였다.

![ReAct ALFWorld results]({{ '/assets/images/blog/react-alfworld-results.png' | relative_url }})

*출처: 논문 Table 3. ReAct는 여러 ALFWorld task에서 강한 성능을 보인다.*

## 11. Fine-tuning 결과

논문은 ReAct를 prompting으로만 쓰지 않고, ReAct trajectory를 이용한 fine-tuning도 실험했다.

![ReAct fine-tuning results]({{ '/assets/images/blog/react-finetuning.png' | relative_url }})

*출처: 논문 Figure 6. Prompting에서는 큰 모델이 유리하지만, ReAct trajectory로 fine-tuning하면 작은 모델도 강해진다.*

핵심 결과는 이렇다.

- prompting만 할 때는 작은 모델이 ReAct 패턴을 잘 따르기 어렵다.
- ReAct trajectory로 fine-tuning하면 작은 모델에서도 ReAct가 가장 좋은 방식이 된다.
- 논문은 PaLM-8B fine-tuned ReAct가 PaLM-62B prompting 방법들을 넘고, PaLM-62B fine-tuned ReAct가 PaLM-540B prompting 방법들을 넘었다고 설명한다.

이 결과는 ReAct를 단순한 prompt trick으로만 보면 안 된다는 점을 보여준다.

> ReAct는 prompt 형식이기도 하지만, trajectory 데이터로 학습시킬 수 있는 agent policy 형식이기도 하다.

## 12. ReAct의 한계

ReAct는 오늘날 agent loop의 원형이지만, 모든 문제를 해결한 것은 아니다.

첫 번째 한계는 prompt 길이다. ReAct 예시는 길다.

```text
Question
Thought
Action
Observation
Thought
Action
Observation
...
Finish
```

복잡한 task에서는 few-shot 예시 몇 개만 넣어도 context length가 빨리 찬다. 논문도 large action space를 가진 복잡한 task에서는 더 많은 demonstration이 필요하지만, input length limit에 부딪힌다고 설명한다.

두 번째 한계는 반복 loop다.

```text
Thought:
Front Row를 검색해야 한다.
Action:
Search[Front Row]
Observation:
검색 실패

Thought:
Front Row를 검색해야 한다.
Action:
Search[Front Row]
Observation:
검색 실패
```

실제 구현에서는 `max_steps`, `repeated_action_detection`, `fallback`이 필요하다.

세 번째 한계는 검색 품질 의존성이다. ReAct는 observation이 좋아야 잘 된다. 검색 결과가 비어 있거나 쓸모없으면 reasoning도 흔들린다.

네 번째 한계는 reflection과 장기 memory가 없다는 점이다. ReAct는 현재 episode 안에서 observation을 보고 다음 action을 고치지만, 실패 후에 "왜 실패했는가?"를 저장하고 다음 episode에서 활용하는 구조는 없다.

다섯 번째 한계는 tool을 스스로 배우지 못한다는 점이다. 논문에서는 `Search`, `Lookup`, `Finish`를 사람이 정해준다. 실제 agent에서는 언제 검색하고, 언제 계산하고, 언제 DB를 조회하고, 언제 코드를 실행할지 훨씬 복잡한 선택이 필요하다.

마지막으로 안전성 문제가 있다. LLM이 action space를 가지면 외부 환경에 영향을 줄 수 있다. 그래서 현대 agent에는 tool permission, human approval, sandbox, rate limit, policy guardrail, audit log가 필요하다.

## 13. 후속 연구와 연결

ReAct 이후 agent 연구는 대부분 ReAct의 빈칸을 채우는 방향으로 발전했다.

| 연구 / 시스템 | ReAct에서 확장한 부분 |
|---|---|
| Reflexion | 실패한 trajectory를 reflection memory로 저장한다 |
| Toolformer | 어떤 상황에서 API를 호출할지 학습한다 |
| Generative Agents | memory stream, reflection, planning을 결합한다 |
| Voyager | 성공한 행동을 skill library로 저장한다 |
| LangGraph / Agents SDK | production workflow로 agent loop를 구조화한다 |

ReAct가 현재 step 안에서 `Observation`을 보고 다음 action을 수정한다면, Reflexion은 실패한 episode를 돌아보고 다음 episode에서 쓸 언어 memory를 만든다. Generative Agents와 Voyager는 이 memory를 더 장기적으로 확장한다.

이 흐름을 한 줄로 쓰면 이렇다.

```text
ReAct
-> Reflection
-> Memory
-> Tool Learning
-> Planning
-> Production Agent Workflow
```

## 14. 내가 이해한 ReAct의 핵심

내가 이해한 ReAct의 가장 중요한 메시지는 "LLM을 정적인 답변 생성기로 보지 말자"는 것이다.

CoT는 정적이다.

```text
질문 -> 내부 추론 -> 답
```

ReAct는 동적이다.

```text
질문 -> 생각 -> 행동 -> 관찰 -> 생각 수정 -> 행동 수정 -> 답
```

이 차이가 크다. ReAct는 LLM을 환경과 상호작용하면서 상태를 업데이트하는 agent로 본다.

그래서 ReAct의 핵심은 다음 세 단어로 요약된다.

```text
Thought -> Action -> Observation
```

하지만 더 정확히는 이렇게 기억하는 편이 좋다.

```text
Thought -> Action -> Observation -> Updated Thought
```

CoT는 생각만 해서 hallucination이 생길 수 있고, Act-only는 행동만 해서 계획을 잘 세우지 못한다. ReAct는 둘을 결합해 더 grounded하고, 해석 가능하고, 수정 가능한 agent 행동을 만든다.

현대 tool-calling agent, LangGraph workflow, OpenAI Agents SDK, MCP 기반 agent를 볼 때도 이 구조를 떠올리면 이해가 훨씬 쉬워진다.
