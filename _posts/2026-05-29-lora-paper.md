---
title: "LoRA 논문 정리: Low-Rank Adaptation으로 효율적인 파인튜닝하기"
date: 2026-05-29 01:10:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - LoRA
  - Paper
source: "arXiv:2106.09685"
---

# LoRA: Low-Rank Adaptation of Large Language Models

이 글은 논문 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)를 바탕으로 정리한 글이다.

논문의 핵심은 간단하다. 큰 사전학습 모델 전체를 fine-tuning하지 말고, 원래 weight는 고정한 채 작은 low-rank 행렬만 학습해도 downstream task에 잘 적응할 수 있다는 것이다. LoRA는 이를 통해 학습 파라미터 수와 GPU 메모리 사용량을 크게 줄이면서도 inference latency를 늘리지 않는 방법을 제안한다.

![LoRA 원문 Figure: 저랭크 재파라미터화]({{ '/assets/images/blog/lora-paper-figure.png' | relative_url }})

*출처: 논문 Figure 1. LoRA는 고정된 pre-trained weight 옆에 학습 가능한 low-rank 행렬 A와 B를 추가한다.*

## 1. 이 논문이 나온 배경

대규모 언어 모델은 downstream task에 맞게 fine-tuning하면 강력하다. 하지만 모델이 커질수록 full fine-tuning의 비용은 빠르게 커진다.

예를 들어 GPT-3 175B 같은 모델을 task마다 full fine-tuning한다고 생각해보자.

- 모든 weight에 대해 gradient와 optimizer state를 저장해야 한다.
- task마다 거대한 fine-tuned checkpoint를 별도로 보관해야 한다.
- 학습에 필요한 GPU 메모리와 통신 비용이 커진다.
- 여러 task를 서비스할 때 모델 교체 비용이 크다.

이 문제를 해결하기 위해 adapter, prefix tuning 같은 parameter-efficient fine-tuning 방법이 제안되었다. 하지만 LoRA 논문은 기존 방법에도 inference latency, context length 감소, 최적화 불안정 같은 한계가 있다고 본다.

## 2. LoRA의 핵심 아이디어

LoRA는 fine-tuning 중 weight 전체를 직접 업데이트하지 않는다.

원래 full fine-tuning은 사전학습 weight $W_0$가 다음처럼 바뀐다고 볼 수 있다.

$$
W = W_0 + \Delta W
$$

LoRA는 이 변화량 $\Delta W$가 낮은 rank로 근사될 수 있다고 가정한다.

$$
\Delta W = BA
$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$이고, $r \ll \min(d, k)$다.

즉 원래 거대한 matrix 전체를 학습하는 대신, 작은 두 행렬 $A$, $B$만 학습한다.

```text
Full fine-tuning:
W 전체를 업데이트

LoRA:
W0는 고정
작은 A, B만 학습
출력은 W0x + BAx
```

![lora overview]({{ '/assets/images/blog/lora.png' | relative_url }})

*LoRA는 pre-trained weight를 고정하고, low-rank update branch만 학습한다.*

## 3. 왜 low-rank가 가능한가?

논문의 가정은 fine-tuning으로 필요한 weight 변화가 전체 parameter space를 모두 쓰지 않을 수 있다는 것이다.

대규모 사전학습 모델은 이미 많은 일반 언어 능력을 갖고 있다. Downstream task fine-tuning은 모델을 처음부터 새로 배우게 하는 것이 아니라, 이미 학습된 표현을 특정 방향으로 조금 이동시키는 과정에 가깝다.

따라서 그 변화량 $\Delta W$가 full-rank일 필요는 없고, 낮은 내재 차원의 subspace에서 표현될 수 있다고 본다.

이 관점은 LoRA의 이름 그대로 `Low-Rank Adaptation`이다.

## 4. 기존 방법과의 차이

LoRA가 비교하는 대표적인 방법은 adapter와 prefix tuning이다.

![lora vs adapter vs prefix tuning]({{ '/assets/images/blog/lorascore.png' | relative_url }})

*Adapter는 모델 내부에 작은 layer를 추가하고, prefix tuning은 입력 앞에 학습 가능한 prefix를 붙이며, LoRA는 weight update를 low-rank matrix로 표현한다.*

| 방법 | 핵심 방식 | 장점 | 한계 |
|---|---|---|---|
| Full fine-tuning | 전체 weight 업데이트 | 표현력 큼 | 비용과 저장공간 큼 |
| Adapter | layer 사이에 작은 module 추가 | 학습 파라미터 적음 | inference latency 증가 가능 |
| Prefix tuning | 입력 prefix vector 학습 | 모델 weight 고정 | context 길이 일부 사용, 최적화 불안정 가능 |
| LoRA | low-rank update 학습 | 파라미터 적고 병합 가능 | 적용할 weight 선택과 rank 설정 필요 |

LoRA의 중요한 장점은 inference 때 $W_0 + BA$를 하나의 weight로 합칠 수 있다는 점이다. 따라서 학습 때는 parameter-efficient하고, 배포 때는 추가 branch 없이 일반 linear layer처럼 실행할 수 있다.

## 5. LoRA는 어디에 적용하는가?

Transformer에는 여러 weight matrix가 있다.

- self-attention의 $W_q$, $W_k$, $W_v$, $W_o$
- feed-forward layer의 projection matrix
- task-specific output head

LoRA 논문은 특히 attention weight에 low-rank update를 적용하는 설정을 실험한다. 모든 matrix를 다 업데이트하지 않아도 충분히 좋은 성능을 낼 수 있는지 보는 것이 핵심이다.

실제 구현에서는 target module을 선택하고 rank $r$, scaling factor, dropout 등을 설정한다.

```text
h = W0 x + BA x
```

여기서 $W_0$는 고정되고, $A$, $B$만 학습된다. 보통 $A$는 random initialization, $B$는 zero initialization을 사용해 학습 시작 시점에는 원래 모델 출력과 같게 만든다.

## 6. 실험 결과 요약

논문은 GPT-2, GPT-3, RoBERTa, DeBERTa 등 여러 모델과 benchmark에서 LoRA를 평가한다.

핵심 결과는 다음과 같다.

- full fine-tuning과 비슷하거나 더 나은 성능을 보인 경우가 있다.
- 학습 가능한 파라미터 수를 크게 줄일 수 있다.
- GPU memory 요구량이 줄어든다.
- adapter 계열과 달리 inference latency를 추가하지 않는다.
- GPT-3 175B 같은 큰 모델에서도 작은 rank로 경쟁력 있는 결과를 보였다.

논문이 강조하는 점은 "학습 파라미터가 적다"만이 아니다. 실제 시스템에서는 checkpoint 저장, task 전환, inference latency도 중요하다. LoRA는 task별로 작은 A/B 행렬만 저장하고, 필요할 때 base model에 병합하거나 분리할 수 있다.

## 7. 한계점

LoRA에도 한계가 있다.

### target module 선택이 필요하다

어떤 weight matrix에 LoRA를 적용할지 선택해야 한다. attention의 query/value에만 적용할지, feed-forward layer까지 적용할지에 따라 성능과 비용이 달라진다.

### rank 선택이 중요하다

rank $r$이 너무 작으면 task에 필요한 변화를 충분히 표현하지 못할 수 있다. 반대로 너무 크면 parameter-efficient fine-tuning의 장점이 줄어든다.

### batch 안에서 여러 LoRA를 섞기 어렵다

서로 다른 요청이 서로 다른 LoRA adapter를 사용해야 하는 serving 환경에서는 batching과 weight switching이 복잡해질 수 있다.

### 모든 task에 항상 충분한 것은 아니다

LoRA는 강력한 fine-tuning 방법이지만, task가 요구하는 변화가 크거나 모델 자체가 부족한 경우에는 full fine-tuning이나 다른 방법이 필요할 수 있다.

## 8. 이 논문의 핵심 의의

LoRA의 의의는 대규모 모델 fine-tuning을 현실적인 비용으로 낮춘 데 있다.

특히 다음 흐름을 만들었다.

- 하나의 base model을 공유하고 task별 작은 adapter만 저장하는 방식
- PEFT(Parameter-Efficient Fine-Tuning)의 실용화
- instruction tuning, domain adaptation, 개인화 모델 학습의 비용 절감
- inference latency를 늘리지 않는 adapter 설계

오늘날 LLM fine-tuning에서 LoRA와 QLoRA 계열이 널리 쓰이는 이유도 이 지점에 있다. 모델 전체를 복사하지 않고도 특정 task나 domain에 맞춘 변화를 학습할 수 있기 때문이다.

## 내가 이해한 핵심

LoRA는 full fine-tuning을 다음처럼 바꾼다.

```text
전체 weight를 다 바꾸지 말고,
weight 변화량만 작은 low-rank 행렬로 학습하자.
```

이 방식은 학습 비용, 저장 비용, task 전환 비용을 줄인다. 그리고 배포 시 low-rank update를 원래 weight에 병합할 수 있어 inference latency를 추가하지 않는다는 점이 크다.

## 한 줄 요약

LoRA는 사전학습 모델 weight를 고정하고 low-rank update 행렬만 학습해, 대규모 언어 모델을 적은 파라미터와 낮은 메모리 비용으로 fine-tuning하는 방법이다.

## 참고 자료

- Edward J. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
- [microsoft/LoRA](https://github.com/microsoft/LoRA)
