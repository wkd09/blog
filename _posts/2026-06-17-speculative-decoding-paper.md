---
title: "Speculative Decoding 논문 정리: 작은 모델로 큰 모델의 디코딩을 빠르게 만들기"
date: 2026-06-17 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - Inference
  - Decoding
  - Paper
source: "arXiv:2211.17192"
---

# Fast Inference from Transformers via Speculative Decoding

이 글은 논문 [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)을 바탕으로 정리한 글이다.

> Yaniv Leviathan, Matan Kalman, Yossi Matias  
> arXiv 2022, ICML 2023 Oral. [[Paper](https://arxiv.org/abs/2211.17192)]

## 1. 이 논문이 다루는 문제

Speculative Decoding은 **작은 근사 모델이 여러 토큰을 먼저 제안하고, 큰 타깃 모델이 그 제안을 한 번에 검증해서 autoregressive decoding의 직렬 병목을 줄이는 방법**이다.

중요한 점은 단순히 빠른 근사 생성을 하는 것이 아니라, 최종 출력 분포가 큰 모델만으로 샘플링했을 때와 같도록 보장한다는 것이다.

## 2. 왜 필요한가

Autoregressive Transformer는 토큰을 하나씩 생성한다.

토큰 $K$개를 생성하려면 큰 모델을 $K$번 순차 실행해야 한다.

```text
prefix -> token 1
prefix + token 1 -> token 2
prefix + token 1 + token 2 -> token 3
...
```

이 구조에서는 GPU에 여유 연산 자원이 있어도 다음 토큰의 입력이 이전 토큰에 의존하기 때문에 병렬화가 어렵다. 특히 큰 LLM은 한 step의 latency가 크고, serving에서는 decode 단계가 사용자 체감 속도에 직접 영향을 준다.

기존 가속 방법들은 모델 구조를 바꾸거나, 재학습이 필요하거나, 출력 품질과 분포가 바뀔 수 있다. 이 논문은 다음 조건을 동시에 만족하려고 한다.

- 이미 학습된 모델을 그대로 사용한다.
- 큰 모델의 출력 분포를 바꾸지 않는다.
- 여러 토큰을 한 번의 target model 검증으로 처리한다.
- memory bandwidth 병목이 있는 환경에서 남는 연산 병렬성을 활용한다.

## 3. 핵심 아이디어

논문에서는 두 모델을 둔다.

- $M_p$: 실제로 쓰고 싶은 큰 target model
- $M_q$: 더 작고 빠른 approximation model

과정은 다음과 같다.

1. 작은 모델 $M_q$가 현재 prefix에서 $\gamma$개의 토큰을 autoregressive하게 미리 생성한다.
2. 큰 모델 $M_p$는 원래 prefix부터, 작은 모델이 제안한 prefix들을 포함한 여러 위치의 다음 토큰 분포를 병렬로 계산한다.
3. 각 제안 토큰을 큰 모델의 분포 기준으로 accept 또는 reject한다.
4. reject가 발생하면 보정된 분포에서 토큰을 다시 샘플링한다.
5. 한 iteration에서 최소 1개, 최대 $\gamma + 1$개의 토큰을 생성한다.

![Speculative decoding에서 작은 모델의 drafting과 큰 모델의 verification 흐름]({{ '/assets/images/blog/specde.png' | relative_url }})

*그림의 왼쪽 패널이 이 글에서 다루는 speculative decoding이다. 작은 모델 $M_q$가 draft token을 순차적으로 만들고, 큰 모델 $M_p$가 여러 후보 위치를 병렬로 검증한다.*

직관적으로는 CPU의 branch prediction과 비슷하다. 작은 모델이 "아마 다음 토큰들은 이럴 것"이라고 예측하고, 큰 모델이 그 예측이 자기 분포와 양립 가능한지 확인한다.

## 4. Speculative Sampling

가장 중요한 부분은 "빠르지만 결과가 바뀌면 안 된다"는 제약이다.

작은 모델의 분포를 $q(x)$, 큰 모델의 분포를 $p(x)$라고 하자. 작은 모델이 토큰 $x$를 샘플링했을 때, 이 토큰은 다음 확률로 accept된다.

$$
\min\left(1, \frac{p(x)}{q(x)}\right)
$$

만약 $q(x) \le p(x)$라면 작은 모델이 제안한 토큰은 항상 accept된다. 작은 모델이 큰 모델보다 해당 토큰을 더 과하게 믿는 경우, 즉 $q(x) > p(x)$일 때는 $p(x) / q(x)$ 비율로만 accept한다.

reject되면 그냥 큰 모델 분포 $p(x)$에서 다시 뽑지 않는다. 대신 다음 보정 분포를 사용한다.

$$
p'(x) = norm(\max(0, p(x) - q(x)))
$$

이 보정이 핵심이다. accept된 영역과 reject 후 재샘플링 영역을 합치면 최종적으로는 $p(x)$에서 직접 샘플링한 것과 같은 분포가 된다.

즉 speculative decoding은 approximate decoding이 아니다. 작은 모델을 쓰지만 최종 샘플링 분포는 target model의 분포를 유지한다.

## 5. 알고리즘 흐름

한 step을 단순화하면 다음과 같다.

```text
1. Mq가 x1, x2, ..., x_gamma를 순차적으로 제안한다.
2. Mp가 prefix, prefix+x1, ..., prefix+x1...x_gamma에 대한 다음 토큰 분포를 병렬 계산한다.
3. 앞에서부터 제안 토큰을 검사한다.
4. accept된 토큰들은 그대로 붙인다.
5. 첫 reject 위치에서는 보정 분포로 새 토큰을 뽑고 iteration을 끝낸다.
6. 전부 accept되면 Mp의 마지막 분포에서 토큰 하나를 추가로 뽑는다.
```

매 iteration은 적어도 하나의 토큰을 생성한다. 따라서 최악의 경우에도 큰 모델을 일반 autoregressive decoding보다 더 많이 순차 호출하지는 않는다. 작은 모델의 제안이 잘 맞을수록 한 번의 큰 모델 호출에서 여러 토큰을 전진할 수 있다.

## 6. 성능을 결정하는 값

논문에서 성능을 이해하는 핵심 변수는 두 가지다.

- $\alpha$: 작은 모델 제안이 accept될 평균 확률
- $c$: 작은 모델 한 step의 wall time이 큰 모델 한 step 대비 얼마나 싼지 나타내는 비율

$\alpha$가 높다는 것은 작은 모델이 큰 모델의 분포를 잘 따라간다는 뜻이다. $c$가 낮다는 것은 작은 모델이 충분히 싸다는 뜻이다.

제안 길이를 $\gamma$라고 하면, 한 iteration에서 생성되는 기대 토큰 수는 다음과 같이 분석된다.

$$
E[\#tokens] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}
$$

wall time 관점의 기대 speedup은 작은 모델 실행 비용까지 포함해 다음처럼 정리된다.

$$
\frac{1 - \alpha^{\gamma + 1}}{(1 - \alpha)(\gamma c + 1)}
$$

![speculative decoding speedup analysis]({{ '/assets/images/blog/speculative-decoding-speedup-analysis.png' | relative_url }})

위 그림은 논문 Figure 3, Figure 4에서 가져온 것이다. 왼쪽은 accept rate인 $\alpha$와 작은 모델 비용 $c$에 따라 최적의 $\gamma$가 어떻게 달라지는지 보여주고, 오른쪽은 $\gamma$를 키울 때 speedup과 연산량 증가가 어떻게 trade-off를 이루는지 보여준다.

이 식에서 볼 수 있듯이, $\gamma$를 무작정 크게 잡는다고 항상 좋아지는 것은 아니다. 작은 모델을 여러 번 돌리는 비용이 있고, reject가 자주 일어나면 speculative computation이 낭비된다.

## 7. 실험 결과

논문은 T5-XXL, LaMDA, GPT-like 모델 등에서 speculative decoding을 평가한다.

핵심 결과는 다음과 같다.

- T5-XXL 기준으로 기존 T5X 구현 대비 약 2배에서 3배 수준의 latency 개선을 보였다.
- 모델 구조 변경이나 재학습 없이 off-the-shelf 모델에 적용했다.
- 출력 분포를 유지하기 때문에 동일한 sampling 설정에서는 target model만 사용하는 경우와 분포적으로 같은 결과를 보장한다.
- 작은 모델은 보통 target model보다 두 자릿수 정도 작은 Transformer가 잘 맞았다.
- 아주 단순한 n-gram 모델도 approximation model로 쓸 수 있지만, accept rate가 낮아 speedup은 제한적이다.

흥미로운 점은 이 방법이 연산량 자체를 항상 줄이는 것은 아니라는 점이다. 오히려 target model을 여러 prefix에 대해 병렬 평가하므로 총 FLOPs는 증가할 수 있다. 대신 decode가 memory bandwidth나 communication에 묶여 있고 병렬 연산 자원이 남아 있는 상황에서는 wall-clock latency가 줄어든다.

## 8. 기존 Decoding과의 차이

일반 decoding은 큰 모델이 매 토큰마다 한 번씩 실행된다.

```text
Mp -> 1 token
Mp -> 1 token
Mp -> 1 token
```

Speculative decoding은 작은 모델이 draft를 만들고, 큰 모델이 한 번에 검증한다.

```text
Mq -> draft tokens
Mp -> verify in parallel
accept several tokens or correct at first reject
```

이 차이는 serving 관점에서 중요하다. 모델의 파라미터와 KV cache를 매 토큰마다 반복적으로 읽는 비용을 줄일 수 있기 때문이다. 즉 이 논문은 decoding을 "계산량 줄이기"보다 "직렬 실행을 병렬 검증으로 바꾸기"에 가깝게 바라본다.

## 9. 한계점

첫 번째 한계는 충분한 병렬 연산 자원이 필요하다는 점이다. target model을 $\gamma + 1$개 prefix에 대해 병렬 평가해야 하므로, GPU가 이미 compute-bound라면 이득이 작거나 사라질 수 있다.

두 번째는 작은 모델 선택이 중요하다는 점이다. 작은 모델이 너무 약하면 accept rate가 낮고, 너무 크면 $c$가 커져서 이득이 줄어든다.

세 번째는 짧은 생성에서는 speedup이 제한된다. 한 iteration에서 적어도 한 번은 큰 모델을 실행해야 하므로, 생성 길이가 짧으면 병렬화 이득을 충분히 회수하기 어렵다.

네 번째는 strict한 distribution 보장을 유지하면 추가적인 근사 최적화 여지가 제한된다. 논문 부록에서는 lenience를 허용하면 더 빨라질 수 있음을 다루지만, 이 경우 "완전히 같은 분포"라는 강한 장점은 약해진다.

## 10. 실제 시스템과의 연결

LLM serving에서 speculative decoding은 decode latency를 줄이는 대표적인 방법이다.

특히 다음 조건에서 잘 맞는다.

- target model이 크고 한 step latency가 크다.
- 작은 draft model을 함께 올릴 수 있다.
- batch나 hardware 상황상 병렬 검증을 수행할 여유가 있다.
- 출력 분포를 바꾸지 않는 가속이 필요하다.

최근 LLM serving 시스템에서는 speculative decoding이 KV cache 관리, continuous batching, paged attention, quantization과 함께 쓰일 수 있다. 다만 이 방법은 attention kernel 자체를 빠르게 만드는 기술이 아니라 decoding algorithm의 직렬성을 줄이는 기술이다.

## 내가 이해한 핵심

이 논문의 핵심은 "작은 모델로 대충 생성한다"가 아니다. 핵심은 작은 모델의 제안을 **큰 모델 분포를 보존하는 방식으로 검증하고 보정한다**는 점이다.

그래서 speculative decoding은 품질을 약간 희생하는 heuristic decoding이라기보다, target model sampling을 더 병렬적으로 실행하기 위한 알고리즘에 가깝다.

또 하나 중요한 점은 speedup의 원천이다. FLOPs를 줄이는 것이 아니라, memory bandwidth와 직렬 decode 병목을 병렬 검증으로 완화한다. 따라서 이 방법의 효과는 모델 크기뿐 아니라 hardware, batch size, draft model 비용, accept rate에 강하게 의존한다.

정리하면 speculative decoding은 다음 문장으로 기억할 수 있다.

> 작은 모델이 길을 먼저 그려 보고, 큰 모델이 그 길을 한 번에 검문한다. 단, 틀린 길은 수학적으로 보정해서 큰 모델만 썼을 때의 분포를 유지한다.

## 한 줄 요약

Speculative Decoding은 작은 모델이 draft token을 만들고 큰 모델이 이를 병렬로 검증해, target model의 출력 분포를 유지하면서 autoregressive decoding의 순차 병목을 줄이는 방법이다.

## 참고 자료

- Yaniv Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
