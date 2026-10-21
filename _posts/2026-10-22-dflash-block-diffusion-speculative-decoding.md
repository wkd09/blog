---
title: "DFlash 정리: Block Diffusion으로 병렬 speculative decoding"
date: 2026-10-22 00:00:00 +0900
summary: "DFlash가 타깃 모델의 hidden feature를 작은 block-diffusion drafter에 주입해 여러 토큰을 병렬 제안하고, 타깃 모델의 검증으로 출력 품질을 보존하는 방법을 살펴본다."
categories:
  - research
tags:
  - AI
  - LLM
  - Inference
  - Speculative Decoding
  - Diffusion
  - Serving
source: "Chen et al., DFlash: Block Diffusion for Flash Speculative Decoding, arXiv:2602.06036v2"
---

LLM은 다음 토큰을 하나씩 생성한다. 각 단계가 이전 단계의 결과를 기다리므로, 답변이 길수록 순차적인 decode 지연이 쌓인다. Speculative decoding은 작은 **draft model**이 다음 토큰들을 미리 제안하고, 큰 **target model**이 한 번에 검증해 이 병목을 줄인다. 결과 토큰은 target model의 분포를 보존하면서 더 빨리 만들 수 있다.

기존의 대표적인 drafter도 토큰을 순서대로 생성한다. 제안 길이를 늘리려면 forward pass를 반복해야 하고, 앞쪽 제안이 틀리면 뒤 토큰도 함께 버려져 draft 비용과 오차 누적이 커진다. 반대로 diffusion 모델은 여러 위치를 병렬로 채울 수 있지만, target과 무관하게 자체 문맥만으로 제안하면 품질과 acceptance가 부족할 수 있다.

[DFlash 논문](https://arxiv.org/abs/2602.06036)은 이 간극을 **target 모델의 문맥 표현을 조건으로 삼는 block diffusion drafter**로 메운다. 작은 draft 모델이 다음 토큰 블록을 한 번의 forward pass로 제안하고, target 모델이 그 블록을 병렬 검증한다. 저자들은 Qwen3 모델 실험에서 최대 6배가 넘는 lossless 가속을 보고한다.

## 핵심 구조: target이 draft에 문맥을 넘긴다

입력 prompt를 받으면 target 모델은 먼저 일반적인 prefill을 수행한다. 이때 여러 층에서 뽑은 hidden state를 결합해 compact한 context feature를 만든다. DFlash는 이 feature를 draft 모델 각 층의 **Key와 Value에 주입**해 KV cache에 보관한다.

![DFlash 추론 구조: target 모델에서 추출한 문맥 feature를 draft layer의 KV에 주입해 block 단위 초안을 만들고 target 모델이 검증한다](/assets/images/blog/dflash-inference-design.svg)

*DFlash 추론 구조. Target에서 얻은 hidden context feature는 draft layer의 KV cache에 들어가고, block diffusion drafter는 이를 바탕으로 여러 토큰을 병렬 제안한다. 출처: [DFlash 논문 Figure 2](https://arxiv.org/html/2602.06036).* 

기존 draft 방식처럼 target feature를 입력 embedding에 한 번 섞고 끝내지 않는다. draft 층이 깊어져도 target 정보가 계속 쓰이도록 각 layer의 KV로 넣는다. 저자들의 설계 의도는 draft가 target의 풍부한 표현을 계속 참고하게 해서, 작은 모델이 문맥을 처음부터 추론하는 부담을 줄이는 것이다.

추론 사이클은 다음과 같다.

1. Target 모델이 현재 prompt와 이미 확정된 prefix를 처리하고 문맥 feature를 제공한다.
2. Block diffusion drafter가 다음 블록 안의 여러 위치를 한 번에 예측한다.
3. Target 모델이 제안 블록을 병렬로 검증한다.
4. 검증된 토큰은 출력에 확정하고, 거부된 위치부터 다음 사이클을 시작한다.

검증을 target 모델이 맡으므로 최종 결과의 분포를 보존할 수 있다. **초안이 병렬이라는 사실만으로 정확도가 보장되는 것은 아니며, 검증 단계가 있어야 lossless speculative decoding이 된다.**

## 왜 block diffusion을 쓰나

Autoregressive drafter는 제안 토큰 수를 늘릴수록 순차 실행 횟수도 늘어난다. DFlash는 블록 안의 mask token들을 병렬로 예측하므로, draft 비용이 토큰 수에 비례해 순차적으로 늘어나는 구조를 피한다. 병렬 처리가 GPU 실행에 더 잘 맞고, drafter를 얕은 한두 층으로 제한하지 않아도 되는 여지가 생긴다.

다만 블록의 모든 위치를 처음부터 무조건 예측하는 단순한 학습은 speculative decoding과 잘 맞지 않는다. 실제 추론에서는 직전 target 검증으로 확정된 토큰이 다음 블록의 시작점 역할을 한다. DFlash 학습은 응답 안에서 **anchor token**을 무작위로 고르고, anchor 뒤의 위치들을 mask해 그 다음 블록을 예측하도록 구성한다. 따라서 학습 시 draft 모델은 target이 확정한 토큰을 출발점으로 삼는 추론 상황을 접한다.

블록 안에서는 양방향 attention으로 mask 위치끼리 정보를 주고받되, 서로 다른 블록 사이의 정보 누출은 attention mask로 막는다. 또한 블록 앞쪽의 예측이 틀리면 뒤쪽 토큰이 함께 무효가 될 수 있으므로, 학습 loss는 앞쪽 위치에 더 큰 가중치를 준다. 이는 평균 토큰 정확도뿐 아니라 실제로 한 사이클에서 받아들여지는 길이를 높이기 위한 설계다.

## 보고된 성능

논문 Table 1은 Qwen3-4B와 Qwen3-8B에서 EAGLE-3와 DFlash를 비교한다. 아래는 Qwen3-8B, temperature 0, draft block 16 기준의 speedup이다.

| Benchmark | EAGLE-3 (tree 16) | DFlash (block 16) |
| --- | ---: | ---: |
| GSM8K | 1.94× | **5.15×** |
| MATH-500 | 1.81× | **6.08×** |
| AIME25 | 1.79× | **5.62×** |
| HumanEval | 1.89× | **5.14×** |
| MBPP | 1.69× | **4.65×** |
| LiveCodeBench | 1.57× | **5.51×** |
| MT-Bench | 1.63× | **2.75×** |
| 평균 | 1.76× | **4.86×** |

이 수치는 일반 autoregressive decoding을 기준으로 한 배수다. 논문은 같은 Qwen3-8B 설정에서 EAGLE-3의 tree 크기를 60으로 키운 경우 평균 2.02×를 보고한다. 따라서 DFlash가 비교 설정에서 더 빠르다는 결과는 분명하지만, speedup은 모델 크기·작업·온도·backend·초안 길이 등에 따라 달라진다. MT-Bench처럼 대화 평가에서는 다른 수학·코드 항목보다 개선 폭이 작다.

논문은 Qwen3-4B, Qwen3-8B 외에도 여러 target 모델 및 serving 설정을 평가하고, SGLang 기반 실험도 제시한다. 다만 표의 speedup을 모든 모델과 하드웨어에서 그대로 기대할 수는 없다. 저자들은 논문 전반에서 6배 초과의 가속 및 EAGLE-3 대비 최대 2.5배 개선을 요약하지만, 이는 각 실험 조건에서 나온 최대치이지 일률적인 서비스 보장치가 아니다.

## 실제 적용에서 따져볼 점

- **추가 drafter 학습이 필요하다.** DFlash는 기존 모델에 단순히 켜는 decoding 옵션이 아니다. Target의 표현을 이용해 block diffusion drafter를 학습하고 추론 경로도 지원해야 한다.
- **모델 결합이 필요하다.** Target의 hidden feature 추출, 층 간 feature fusion, draft KV 주입이 구현되어야 한다. 서로 다른 target과 drafter 사이의 호환성이 자동으로 성립하지 않는다.
- **속도 향상은 acceptance와 실행 비용의 합으로 결정된다.** 초안이 길어도 target이 많이 거부하거나 draft 실행·검증 오버헤드가 크면 end-to-end 개선은 줄어든다.
- **평가 조건을 함께 봐야 한다.** 논문 결과는 Qwen3 계열과 지정된 backend 및 benchmark 조건에 기반한다. 실제 서비스에서는 batch 크기, 출력 길이, GPU, 동시성, sampling 설정을 반영해 측정해야 한다.
- **lossless의 범위를 구분해야 한다.** 검증 기반 speculative decoding은 target의 생성 분포를 보존할 수 있다는 뜻이다. 이 표현이 draft가 모든 토큰을 맞히거나 latency가 항상 개선된다는 의미는 아니다.

## 논문 리뷰

### 강점

가장 설득력 있는 부분은 **속도와 초안 품질을 따로 최적화하지 않고, target 표현의 KV 주입으로 함께 다룬 점**이다. Block diffusion의 병렬성만 가져오면 drafter가 target 모델만큼 문맥을 이해하지 못해 acceptance가 낮아질 수 있다. DFlash는 target의 여러 층 표현을 draft 각 층에 지속해서 제공하는 설계로 이 약점을 직접 겨냥한다. Anchor 기반 학습과 앞 위치에 큰 loss 가중치를 주는 것도 speculative decoding에서 초반 토큰이 특히 중요한 이유와 맞닿아 있다.

평가도 단일 수치에만 기대지 않는다. 저자들은 여러 Qwen 계열 모델과 수학·코드·대화 benchmark를 보고하고, SGLang serving 결과와 block 크기, hidden feature 수 등의 ablation을 포함한다. 특히 동시 요청이 늘면 EAGLE-3의 큰 draft tree가 이점을 잃을 수 있는 반면 DFlash가 비교적 높은 처리량 배수를 유지한 실험은, 단순 acceptance 길이보다 serving에서의 전체 비용을 봐야 한다는 점을 보여준다.

### 한계와 해석

첫째, 큰 speedup은 특정 조건에 의존한다. 논문 주요 표의 Qwen3 비교는 Transformers backend와 thinking mode 비활성화 조건이며, 별도 SGLang 실험은 LLaMA-3.1-8B-Instruct, B200 단일 GPU 설정을 사용한다. 예를 들어 그 serving 표에서 DFlash의 speedup은 동시성이 1일 때보다 32일 때 낮아진다. 그러므로 표의 4.86× 평균을 서비스 트래픽에 그대로 대입할 수 없다.

둘째, 실용성의 핵심은 drafter 학습과 시스템 통합이다. Target feature를 추출해 저장하고 draft layer의 KV로 투영해야 하며, target 모델이나 backend가 바뀌면 다시 검증이 필요하다. 논문은 이 비용을 평가하지만, 운영 환경에서 모델 교체 주기와 학습·저장 비용까지 포함한 총소유비용은 독자가 별도로 계산해야 한다.

셋째, 평가 범위는 넓어졌지만 보편성의 증명은 아니다. 모델과 task가 다양한 것은 장점이지만, latency와 throughput은 GPU 세대, batch 구성, scheduling overlap, context 길이, 샘플링에 민감하다. 또한 논문이 강조하는 “lossless”는 target 분포를 유지하는 검증 방식의 성질이지, 사용자 체감 지연이나 모든 출력 길이에서 같은 배수의 속도 향상을 보장하지 않는다.

### 종합 의견

DFlash는 **block diffusion을 speculative decoding에 붙이는 데 필요한 조건화 방식을 구체적인 추론 구조와 학습 절차로 연결하고, 강한 실험 결과를 제시한 논문**이다. 기여의 중심은 diffusion 자체보다 target hidden feature를 각 draft layer KV로 유지하는 설계와 추론에 맞춘 block 학습이다. 재현·도입을 판단할 때는 대표 speedup만 인용하기보다, 동일한 backend와 동시성에서 acceptance 길이, end-to-end latency, 처리량, drafter 운영 비용을 함께 비교해야 한다.

## 정리

DFlash의 핵심은 **target 모델이 이미 계산한 내부 문맥을 작은 diffusion drafter에 지속적으로 전달하는 것**이다. Draft 모델은 블록의 다음 토큰들을 병렬로 제안해 순차 draft의 병목을 줄이고, target 모델은 이를 검증해 최종 생성 분포를 보존한다.

실험 결과는 Qwen3의 수학·코드 benchmark에서 특히 높은 speedup을 보인다. 실제 시스템에 적용할 때는 논문의 최대 배수보다 draft 학습 비용, target feature 연결, acceptance 길이, 사용 하드웨어에서 측정한 end-to-end 지연을 함께 확인해야 한다.

## 참고 자료

- Chen et al., [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), arXiv:2602.06036v2, 2026.
