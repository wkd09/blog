---
title: "OpenAI 연구자들은 어디까지 에이전트에 맡길까?"
date: 2026-11-20 00:00:00 +0900
summary: "OpenAI에서 에이전트가 사람의 추가 개입 없이 과제를 끝낸 비율은 훈련 모니터링에서 79%, 훈련 실행에서 35%였다. 에이전트 가동 시간과 연구자당 실험 수도 함께 살펴본다."
categories:
  - research
tags:
  - AI
  - Agent
  - OpenAI
source: "OpenAI, Research acceleration: The jagged frontier (2026-11-18)"
---

OpenAI 연구자 한 명이 하루 동안 가동한 에이전트의 시간은 2026년 5월부터 10월 사이 평균 4시간에서 약 46시간으로 늘었다. 하루가 24시간인데 어떻게 46시간이 나올까? 여러 에이전트가 동시에 일한 시간을 각각 더한 값이다. 아래 그래프는 가동 시간 8시간을 하루로 환산해, 같은 변화를 0.5일에서 5.7일로 표시한다.

![연구자 한 명당 하루 에이전트 가동 시간. 8시간을 하루로 환산한 전체 평균이 0.5일에서 5.7일로 늘었다.]({{ '/assets/images/blog/openai-research-acceleration-2026-11/02-agent-workdays.png' | relative_url }})

10월 중순에는 연구자 3명 중 2명 이상이 하루 최대 동시 가동 수를 기준으로 평균 4개 이상의 에이전트를 썼다.

![동시에 작동한 에이전트 수에 따른 연구자 비율. 10월에는 4개 이상을 사용한 집단이 전체의 3분의 2를 넘었다.]({{ '/assets/images/blog/openai-research-acceleration-2026-11/03-agent-concurrency.png' | relative_url }})

## 에이전트를 얼마나 썼나?

10월 연구자 한 명의 하루 사용량을 API 정가로 환산하면 중앙값은 2,400달러, 90백분위는 23,000달러였다. 두 그래프는 세로축의 금액 범위가 다르다.

<figure class="half">
  <img src="{{ '/assets/images/blog/openai-research-acceleration-2026-11/01a-median-spend.png' | relative_url }}" alt="연구자당 하루 에이전트 사용액의 중앙값">
  <img src="{{ '/assets/images/blog/openai-research-acceleration-2026-11/01b-p90-spend.png' | relative_url }}" alt="연구자당 하루 에이전트 사용액의 90백분위">
  <figcaption>같은 기간의 28일 평균 사용액을 중앙값과 90백분위 순서로 표시했다.</figcaption>
</figure>

9월 연구 조직 전체의 에이전트 사용량을 같은 방식으로 환산한 금액은 주식 보상을 제외한 인건비의 약 2배였다.

![팀별 에이전트 사용액을 주식 보상 제외 인건비로 나눈 비율]({{ '/assets/images/blog/openai-research-acceleration-2026-11/04-agent-spend-vs-labor.png' | relative_url }})

## 훈련을 지켜보는 일과 시작하는 일

8월 1일부터 10월 18일까지, 에이전트가 **사람의 추가 개입 없이 과제를 끝낸 비율**은 훈련 중 이상을 감시할 때 79%, 훈련을 시작할 때 35%로 추정됐다. 모델이 응답을 생성하고 제공하는 코드를 작성할 때는 25%였다. 인터뷰에 참여한 한 연구자는 비슷한 평가 과제를 여럿 맡길 때 먼저 하나만 만들어 보게 했다. 첫 결과를 직접 검토하고 수정한 뒤 나머지를 맡겼다.

![연구 과제별 무개입 성공률. 훈련 모니터링이 가장 높았고 추론·서빙 코드 작성이 가장 낮았다.]({{ '/assets/images/blog/openai-research-acceleration-2026-11/08-success-by-task.png' | relative_url }})

전체 27개 과제를 펼치면, 기본 그래프에서 생략한 중간 순위도 보인다. 예를 들어 연구 결과 정리는 14위, 합성 데이터 생성은 16위였다.

![모니터링부터 추론·서빙 코드 작성까지 27개 과제 각각의 무개입 성공률과 추정 구간]({{ '/assets/images/blog/openai-research-acceleration-2026-11/08b-success-all-tasks.png' | relative_url }})

업무를 큰 범주로 묶었을 때의 무개입 성공률은 의사소통 69%, 분석 58%, 개발 43%였다.

![의사결정·설계·개발·실행·분석·의사소통 범주별 무개입 성공률]({{ '/assets/images/blog/openai-research-acceleration-2026-11/07-success-by-category.png' | relative_url }})

사람이 하면 2~4시간 걸릴 것으로 추정된 과제에서도 무개입 성공률은 3월 23%에서 10월 46%로 올랐다. 그동안 연구자가 골라 맡긴 과제도 달라졌을 수 있어, 상승분 중 얼마가 모델의 성능 향상 덕분인지는 따로 측정해야 한다.

![사람의 예상 작업 시간에 따른 월별 무개입 성공률과 위임된 과제의 비중]({{ '/assets/images/blog/openai-research-acceleration-2026-11/06-success-by-duration.png' | relative_url }})

연구자가 무엇을 자주 맡겼는지도 달라졌다. 아래 그래프에서 면적은 연구자 한 명이 하루에 맡긴 과제 수다. 10월에는 기술 질문에 답하거나 작업을 검토하는 등 의사소통에 해당하는 일이 가장 큰 비중을 차지했다.

![연구자 한 명이 하루에 에이전트에 맡긴 과제 수를 업무 종류별로 쌓아 표시한 그래프]({{ '/assets/images/blog/openai-research-acceleration-2026-11/09-tasks-per-researcher.png' | relative_url }})

## 사람의 시간은 어디로 갔나?

1월부터 9월까지 안전·정렬 연구에 투입된 것으로 추정한 작업 주수는 77% 늘었다. 연구 분야 중 사람이 들인 시간이 절대량으로 가장 많이 늘어난 곳이다.

![연구 분야별 추정 투입 시간의 변화. 안전·정렬 분야는 1월 대비 9월 77% 증가했다.]({{ '/assets/images/blog/openai-research-acceleration-2026-11/05-researcher-time.png' | relative_url }})

이 추정치는 내부 문서와 코드 기여 등 활동 기록에 근거한다. 아래 그림은 한 달에 관측된 연구 과제가 다음 달 기록에도 나타난 비율을 보여준다.

![이전 달에 관측된 연구 과제가 다음 달에도 기록에 나타나는 비율과 다른 선정 기준의 비교]({{ '/assets/images/blog/openai-research-acceleration-2026-11/05b-methods-workstream-persistence.png' | relative_url }})

연구자 101명에게 최신 코딩 에이전트를 3개월간 다시 쓰기 위해 어느 정도의 팀 자원을 내줄 의향이 있는지 물었다. 인력과 연산 자원을 따로 물었을 때 응답의 중앙값은 각각 40%와 30%였다. 2027년 1월을 가정한 연산 자원 질문에서는 50%였다.

<figure class="third">
  <img src="{{ '/assets/images/blog/openai-research-acceleration-2026-11/10a-survey-headcount.png' | relative_url }}" alt="접근권과 교환할 팀 인력 비율의 응답 분포, 중앙값 40%">
  <img src="{{ '/assets/images/blog/openai-research-acceleration-2026-11/10b-survey-compute.png' | relative_url }}" alt="현재 팀 연산 자원 비율의 응답 분포, 중앙값 30%">
  <img src="{{ '/assets/images/blog/openai-research-acceleration-2026-11/10c-survey-future-compute.png' | relative_url }}" alt="2027년 1월을 가정한 연산 자원 비율의 응답 분포, 중앙값 50%">
  <figcaption>막대는 각 구간을 고른 응답자의 가중 비율이고, 점선은 중앙값이다.</figcaption>
</figure>

## 실험은 얼마나 늘었나?

2026년에는 기여자 한 명이 변경한 코드 줄 수가 빠르게 늘었다. 아래 그래프는 2025년 이전 평균을 1로 두고 28일 이동평균을 표시한다.

![활동 기여자 한 명당 변경한 코드 줄 수를 2025년 이전 평균과 비교한 그래프]({{ '/assets/images/blog/openai-research-acceleration-2026-11/11-lines-of-code.png' | relative_url }})

10월 18일까지 4주 동안, 실험을 수행한 연구자 한 명당 기록된 실험 수는 **2025년 기준의 1.68배**였다. 연구자들이 더 많은 실험을 돌렸다는 뜻이다. 그중 어떤 실험이 다음 모델의 성능을 끌어올렸고 에이전트가 거기에 얼마나 기여했는지는 이 그래프만으로 알 수 없다.

![실험 수행자 한 명당 기록된 실험 수. 2025년 기준을 1로 두었을 때 10월 18일까지 4주간 평균은 1.68이다.]({{ '/assets/images/blog/openai-research-acceleration-2026-11/12-research-experiments.png' | relative_url }})

OpenAI는 모델 성능도 별도로 추적했다. 2025년 8월 이후 최고점을 경신한 OpenAI 모델 9개를 기준으로 보면, 내부 성능 지수(OECI)는 연 16.4점씩 오르는 추세였다.

![모델별 OpenAI 성능 지수와 최고점을 경신한 모델 9개의 추세선]({{ '/assets/images/blog/openai-research-acceleration-2026-11/13-capability-index.png' | relative_url }})

이 지수는 여러 벤치마크를 종합한다. 아래 그림에서 세로 위치는 각 벤치마크의 추정 난이도, 원의 면적은 난이도 추정에 사용한 모델 수를 나타낸다.

![평가 범주별 벤치마크의 추정 난이도와 사용된 모델 수]({{ '/assets/images/blog/openai-research-acceleration-2026-11/13b-methods-benchmark-difficulty.png' | relative_url }})

글에 인용한 수치와 그래프는 OpenAI가 2026년 11월 18일 공개한 내부 연구 활동 보고서에서 가져왔다. [OpenAI 원문](https://openai.com/index/research-acceleration-jagged-frontier/)

## 내 생각

훈련 모니터링은 79%, 훈련 실행은 35%라는 결과를 보면서 ‘훈련을 에이전트에 맡긴다’는 말이 꽤 모호하다고 느꼈다. 나라면 여러 실험의 로그를 훑고 이상한 구간을 찾는 일부터 맡기겠다. 학습률을 바꿔 가며 실험을 열 번 돌린다면, 첫 실행에서 학습률만 달라졌는지 직접 확인한 뒤 나머지 아홉 번을 맡기고 싶다.

다음 보고서에서 궁금한 건 연구자가 가설 하나를 검증하는 데 걸린 시간이다. 가능성이 낮은 아이디어를 하루 일찍 접고 다음 실험으로 넘어갈 수 있었다면, 실험 횟수가 그대로여도 연구는 그만큼 빨라졌다고 생각한다.
