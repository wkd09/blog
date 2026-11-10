---
title: "GPTQ 논문 정리: 거대 언어 모델을 3-4bit로 양자화하기"
date: 2026-07-07 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - Quantization
  - GPTQ
  - Inference
  - Paper
source: "arXiv:2210.17323"
---

# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

이 글은 논문 [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)를 바탕으로 정리한 글이다.

> Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
> ICLR 2023. [[Paper](https://arxiv.org/pdf/2210.17323)] [[Code](https://github.com/IST-DASLab/gptq)]

논문의 핵심은 간단하다.

> 거대 언어 모델을 다시 학습하지 않고, weight만 3-4bit로 줄이되, 단순 반올림보다 훨씬 정확하게 만들자.

GPTQ는 `post-training quantization`, 줄여서 `PTQ` 방법이다. 즉 원래 모델을 다시 학습하거나 fine-tuning하지 않는다. 대신 작은 calibration data를 모델에 통과시켜 각 layer의 입력 통계를 모으고, 그 정보를 이용해 weight를 양자화한다.

한 줄로 요약하면 이렇다.

> GPTQ는 layer output이 최대한 덜 변하도록 Hessian 기반 보정을 넣어 weight를 한 번에 3-4bit로 압축하는 LLM용 PTQ 알고리즘이다.

![GPTQ perplexity comparison]({{ '/assets/images/blog/gptq-figure1-perplexity.png' | relative_url }})

*논문 Figure 1. RTN은 낮은 bit에서 perplexity가 크게 무너지는 반면, GPTQ는 FP16 기준선에 훨씬 가깝게 유지된다.*

## 1. 왜 GPTQ가 필요한가

LLM inference에서 가장 먼저 부딪히는 벽은 GPU 메모리다.

예를 들어 175B 파라미터 모델을 FP16으로 저장하면 weight만 대략 350GB 전후가 필요하다. 여기에 KV cache, activation, runtime overhead까지 붙으면 단일 GPU에는 절대 들어가지 않는다.

그래서 큰 모델을 실제로 서빙하려면 보통 다음 중 하나가 필요하다.

- 여러 GPU에 모델을 나누어 올린다.
- 더 작은 모델을 사용한다.
- weight precision을 낮춰 모델 크기를 줄인다.
- activation, KV cache, attention kernel 등 다른 병목도 같이 최적화한다.

이 중 quantization은 매우 실용적인 선택이다. FP16 weight를 4bit로 줄이면 weight 저장량만 놓고 보면 약 4배 작아진다. 3bit면 더 작아진다.

하지만 문제는 정확도다.

단순히 weight를 가장 가까운 quantization level로 반올림하는 `RTN`, 즉 round-to-nearest 방식은 8bit에서는 꽤 잘 작동할 수 있다. 하지만 4bit, 3bit로 내려가면 작은 오차가 layer를 지나며 누적되고, 모델 출력이 크게 흔들릴 수 있다.

GPTQ 논문이 보여주는 가장 강한 메시지는 이 지점이다.

```text
8bit weight quantization:
단순 반올림도 대체로 괜찮을 수 있음

3-4bit weight quantization:
단순 반올림만으로는 불안정함
양자화 오차를 보정하는 알고리즘이 필요함
```

## 2. Quantization을 어떤 문제로 보는가

GPTQ는 layer-wise quantization 문제를 푼다.

어떤 linear layer의 weight를 `W`, calibration input을 `X`라고 하자. 원래 layer output은 다음과 같다.

$$
Y = WX
$$

양자화된 weight를 $\hat{W}$라고 하면, 목표는 원래 output과 양자화 후 output이 최대한 비슷해지도록 만드는 것이다.

$$
\arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2
$$

즉 GPTQ는 weight 자체의 값이 원래 weight와 얼마나 가까운지만 보지 않는다. 더 중요한 것은 **그 weight가 실제 calibration input에 대해 만들어내는 layer output이 얼마나 유지되는가**다.

이 차이가 중요하다.

예를 들어 어떤 weight 하나가 조금 바뀌어도 실제 입력에서 거의 쓰이지 않는 방향이면 영향이 작을 수 있다. 반대로 값 차이는 작아도 자주 활성화되는 입력 방향과 강하게 연결되어 있으면 output 차이가 커질 수 있다.

그래서 GPTQ는 input 통계에서 얻은 Hessian 정보를 사용한다.

논문에서 row 하나를 기준으로 보면 Hessian은 대략 다음 형태다.

$$
H = 2XX^T
$$

여기서 `X`는 calibration input이다. GPTQ는 이 Hessian의 inverse 정보를 이용해, 어떤 weight를 양자화했을 때 남은 weight들을 어떻게 보정해야 layer output 오차가 줄어드는지 계산한다.

## 3. RTN은 왜 약한가

RTN은 매우 단순하다.

```text
각 weight를 가장 가까운 quantization grid 값으로 보낸다.
끝.
```

장점은 빠르다는 것이다. model size가 수백 billion parameter여도 각 weight를 독립적으로 반올림하면 되므로 scale이 잘 된다.

하지만 단점도 명확하다. 각 weight를 독립적으로 처리하기 때문에, 하나의 weight에서 생긴 error를 다른 weight가 보상할 기회를 주지 않는다.

간단한 예를 들어보자.

```text
원래 output:
y = w1*x1 + w2*x2

w1을 양자화하면서 값이 조금 커졌다면,
w2를 약간 줄여서 전체 y를 비슷하게 맞출 수 있을지도 모른다.

RTN:
w1, w2를 각각 독립적으로 반올림한다.

GPTQ:
w1을 양자화한 뒤, 그 오차를 보고 아직 양자화하지 않은 w2 쪽을 보정한다.
```

이것이 GPTQ의 직관이다.

> 양자화는 weight 하나하나의 반올림 문제가 아니라, layer output을 유지하는 reconstruction 문제다.

## 4. OBQ에서 GPTQ로

GPTQ는 완전히 뜬금없이 나온 방법은 아니다. 논문은 `Optimal Brain Quantization`, 줄여서 `OBQ`를 기반으로 한다.

OBQ는 weight를 하나씩 양자화하면서, 양자화로 생긴 오차를 아직 양자화하지 않은 weight에 분산해 보정한다. 핵심 수식은 다음 형태다.

$$
\delta_F =
-\frac{w_q - quant(w_q)}{[H_F^{-1}]_{qq}}
(H_F^{-1})_{:,q}
$$

여기서 의미는 다음과 같다.

| 기호 | 의미 |
|---|---|
| $w_q$ | 지금 양자화할 weight |
| $quant(w_q)$ | quantization grid로 보낸 값 |
| $F$ | 아직 full precision으로 남아 있는 weight 집합 |
| $H_F^{-1}$ | 남은 weight들에 대한 inverse Hessian |
| $\delta_F$ | 남은 weight들에 적용할 보정량 |

직관적으로는 이렇다.

```text
1. weight 하나를 quantize한다.
2. 그 때문에 생긴 error를 계산한다.
3. inverse Hessian을 이용해 남은 weight들을 업데이트한다.
4. 다음 weight로 넘어간다.
```

문제는 OBQ가 너무 느리다는 것이다. 작은 vision model에서는 가능하지만, GPT/OPT/BLOOM 같은 수십B-수백B 파라미터 모델에는 그대로 적용하기 어렵다.

GPTQ의 기여는 여기 있다.

> OBQ의 second-order 보정 아이디어는 유지하되, LLM 크기에서도 몇 시간 안에 돌도록 알고리즘을 바꾼다.

## 5. GPTQ의 핵심 아이디어

논문은 GPTQ가 OBQ를 LLM에 적용 가능하게 만들기 위해 세 가지 변경을 한다.

### 5.1 Greedy order를 버린다

OBQ는 매번 "지금 quantize했을 때 error가 가장 작은 weight"를 고르는 greedy order를 사용한다.

이 방식은 정확도 관점에서는 자연스럽지만, 계산량이 크다. row마다 quantization 순서가 달라질 수 있고, Hessian inverse 업데이트도 훨씬 복잡해진다.

GPTQ의 관찰은 이것이다.

> 큰 layer에서는 꼭 greedy order를 쓰지 않아도 결과 차이가 크지 않다.

그래서 GPTQ는 모든 row에서 같은 순서로 column을 양자화한다. 이렇게 하면 남아 있는 weight 집합이 모든 row에서 같아지고, Hessian inverse 업데이트를 row마다 따로 할 필요가 줄어든다.

복잡도도 크게 줄어든다.

```text
OBQ:
O(d_row * d_col^3)

GPTQ fixed order:
O(max(d_row * d_col^2, d_col^3))
```

LLM의 linear layer는 행렬이 매우 크기 때문에 이 차이가 수십 배가 아니라 수천 배 차이로 이어질 수 있다.

### 5.2 Lazy batch update를 사용한다

두 번째 문제는 GPU 효율이다.

이론적 계산량이 줄어도, 구현이 작은 matrix/vector update를 계속 반복하면 GPU를 잘 쓰지 못한다. 메모리에서 큰 행렬을 읽고 쓰는 데 비해 실제 연산량이 작아서 memory bandwidth 병목이 된다.

GPTQ는 column을 하나씩 처리하되, global update를 매번 하지 않는다. 대신 block 단위로 묶는다.

```text
1. B개 column으로 이루어진 block을 잡는다.
2. block 내부에서는 column-by-column으로 양자화하고 보정한다.
3. block 처리가 끝났을 때 남은 전체 weight에 한 번에 update를 적용한다.
```

논문에서는 기본 block size로 `B = 128`을 사용한다.

이 방식은 이론적 연산량 자체를 크게 줄이는 것보다는 GPU에서 실행되는 형태를 좋게 만든다. 작은 update를 계속 날리는 대신, 더 큰 matrix 연산으로 묶어 GPU utilization을 높인다.

![GPTQ quantization procedure]({{ '/assets/images/blog/gptq-figure2-procedure.png' | relative_url }})

*논문 Figure 2. GPTQ는 block 단위로 weight column을 양자화하고, inverse Hessian 정보를 이용해 아직 양자화되지 않은 weight를 갱신한다.*

### 5.3 Cholesky 형태로 안정화한다

세 번째 문제는 numerical stability다.

Hessian inverse를 반복해서 업데이트하다 보면 큰 모델에서는 수치 오차가 쌓인다. 특히 block update와 함께 쓰면 inverse Hessian이 안정적이지 않아지고, 잘못된 방향으로 weight를 크게 보정할 수 있다.

GPTQ는 이 문제를 Cholesky decomposition으로 다룬다.

논문에서는 dampening을 추가한 inverse Hessian을 다음처럼 만든다.

$$
H^{-1} = (2XX^T + \lambda I)^{-1}
$$

그리고 이 값을 Cholesky 형태로 바꿔, quantization 과정에서 필요한 row 정보를 안정적으로 사용한다. 여기서 $\lambda$는 평균 diagonal 값의 1%로 설정한다.

이 변경은 단순한 구현 디테일처럼 보이지만, 큰 모델에서는 중요하다. 논문은 수십B 이상 모델에서 수치 문제가 실제로 발생한다고 보고한다.

## 6. 전체 알고리즘 흐름

GPTQ의 전체 흐름을 단순화하면 다음과 같다.

```text
입력:
- full precision weight W
- calibration input X
- block size B
- quantization grid

1. H = 2XX^T + lambda I 계산
2. H^{-1} 계산
3. H^{-1}을 Cholesky 형태로 변환
4. weight matrix를 column block으로 나눔
5. 각 block에 대해:
   5.1 column 하나를 quantize
   5.2 quantization error 계산
   5.3 block 내부의 아직 남은 column을 update
   5.4 block이 끝나면 나머지 전체 weight를 lazy update
6. quantized weight Q 반환
```

핵심은 `quantize -> error 계산 -> 남은 weight 보정`이 반복된다는 점이다.

RTN과 비교하면 다음처럼 볼 수 있다.

| 항목 | RTN | GPTQ |
|---|---|---|
| 방식 | weight별 독립 반올림 | layer output 보존을 위한 보정 |
| calibration data | 필요 없음 | 필요 |
| second-order 정보 | 사용 안 함 | inverse Hessian 사용 |
| 속도 | 매우 빠름 | RTN보다 느리지만 수백B 모델도 수 시간 |
| 낮은 bit 안정성 | 3-4bit에서 취약 | 3-4bit에서 훨씬 안정적 |

## 7. 실험 세팅

논문은 OPT와 BLOOM model family 전체를 대상으로 실험한다.

주요 세팅은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 모델 | OPT family, BLOOM family |
| quantization | weight-only PTQ |
| calibration data | C4에서 random 2048-token segment 128개 |
| GPU | 단일 NVIDIA A100 80GB로 quantization |
| baseline | FP16, RTN |
| 주요 metric | WikiText2/PTB/C4 perplexity, LAMBADA accuracy |

여기서 calibration data가 매우 작다는 점이 중요하다. 전체 학습 데이터가 아니라, 일반 텍스트 일부만 사용한다. 또한 task-specific data를 쓰지 않기 때문에 실험은 사실상 zero-shot 성격을 유지한다.

## 8. 결과 1: OPT에서 RTN은 무너지고 GPTQ는 버틴다

OPT family의 WikiText2 perplexity 결과는 GPTQ의 효과를 가장 잘 보여준다.

![OPT WikiText2 table]({{ '/assets/images/blog/gptq-table3-opt-wikitext2.png' | relative_url }})

*논문 Table 3. OPT family의 WikiText2 perplexity. 낮을수록 좋다.*

가장 중요한 숫자는 OPT-175B다.

| 설정 | WikiText2 PPL |
|---|---:|
| FP16 | 8.34 |
| RTN 4bit | 10.54 |
| GPTQ 4bit | 8.37 |
| RTN 3bit | 7.3e3 |
| GPTQ 3bit | 8.68 |

4bit GPTQ는 FP16과 거의 같다. 8.34에서 8.37로 올라갈 뿐이다. 반면 4bit RTN은 10.54까지 나빠진다.

3bit에서는 차이가 훨씬 극단적이다. RTN은 perplexity가 수천 단위로 튄다. 사실상 모델이 망가졌다고 봐야 한다. 반면 GPTQ 3bit는 8.68로, FP16보다 나빠지긴 하지만 여전히 쓸 만한 수준을 유지한다.

흥미로운 점은 작은 모델보다 큰 모델에서 GPTQ가 더 잘 버티는 경향이 있다는 것이다. 논문도 큰 모델이 quantization에 더 robust해 보인다고 해석한다. 직관적으로는 overparameterization이 큰 모델일수록 weight perturbation을 흡수할 여지가 더 많기 때문으로 볼 수 있다.

다만 OPT-66B는 예외적으로 튀는 결과가 있다. 논문은 이 모델의 early layer에 dead unit 비율이 높아 compress하기 어려웠을 가능성을 언급한다.

## 9. 결과 2: BLOOM에서도 같은 패턴

BLOOM family에서도 패턴은 비슷하다.

![BLOOM WikiText2 table]({{ '/assets/images/blog/gptq-table4-bloom-wikitext2.png' | relative_url }})

*논문 Table 4. BLOOM family의 WikiText2 perplexity.*

BLOOM-176B의 핵심 숫자는 다음과 같다.

| 설정 | WikiText2 PPL |
|---|---:|
| FP16 | 8.11 |
| RTN 4bit | 8.37 |
| GPTQ 4bit | 8.21 |
| RTN 3bit | 571 |
| GPTQ 3bit | 8.64 |

BLOOM은 OPT보다 RTN 4bit가 덜 나쁘다. 하지만 3bit에서는 RTN이 여전히 크게 무너진다.

즉 GPTQ의 실질적 가치는 4bit에서도 있지만, 특히 3bit처럼 더 aggressive한 compression regime에서 강하게 드러난다.

## 10. 결과 3: 175B급 모델 상세 비교

논문은 OPT-175B와 BLOOM-176B를 따로 모아 WikiText2, PTB, C4, LAMBADA 결과를 비교한다.

![Large model summary table]({{ '/assets/images/blog/gptq-table5-large-models.png' | relative_url }})

*논문 Table 5. OPT-175B와 BLOOM-176B 상세 결과. PPL은 낮을수록 좋고, LAMBADA는 높을수록 좋다.*

OPT-175B에서 GPTQ 4bit는 거의 FP16 수준이다.

| Metric | FP16 | GPTQ 4bit | GPTQ 3bit |
|---|---:|---:|---:|
| WikiText2 PPL | 8.34 | 8.37 | 8.68 |
| PTB PPL | 12.01 | 12.26 | 12.68 |
| C4 PPL | 10.13 | 10.28 | 10.67 |
| LAMBADA acc | 75.59 | 76.80 | 76.19 |

LAMBADA에서는 GPTQ가 FP16보다 높게 나오는 값도 있다. 이것을 "양자화하면 모델이 더 좋아진다"로 읽으면 위험하다. zero-shot benchmark에는 noise가 있고, quantization이 일종의 regularization처럼 작동하는 경우도 있다. 중요한 것은 성능이 크게 무너지지 않는다는 점이다.

또 하나 중요한 결과는 group size다.

논문은 3bit GPTQ에 `g1024`, `g128` 같은 grouping을 추가하면 정확도가 더 좋아진다고 보여준다. group size가 작아질수록 scale/zero point를 더 세밀하게 잡을 수 있어서 quantization error가 줄어든다. 대신 group별 metadata가 늘어나므로 실제 평균 bit 수는 조금 증가한다.

## 11. 결과 4: LAMBADA zero-shot

Perplexity는 language modeling 품질을 민감하게 보는 지표다. 하지만 실제 task accuracy도 중요하다.

논문은 LAMBADA accuracy도 비교한다.

![LAMBADA accuracy]({{ '/assets/images/blog/gptq-figure3-lambada.png' | relative_url }})

*논문 Figure 3. LAMBADA accuracy에서도 GPTQ는 RTN보다 안정적이다.*

여기서도 같은 결론이 나온다.

- 4bit에서는 GPTQ와 RTN 모두 어느 정도 버티지만 GPTQ가 더 안정적이다.
- 3bit에서는 RTN이 크게 무너진다.
- GPTQ 3bit는 큰 모델에서 꽤 높은 accuracy를 유지한다.

특히 OPT-175B는 3bit GPTQ에서도 LAMBADA 76.19를 기록한다. FP16의 75.59와 비교해도 거의 차이가 없다.

## 12. 결과 5: 실제 inference 메모리와 속도

GPTQ는 단순히 파일 크기를 줄이는 논문이 아니다. 논문은 실제 generative inference에서 latency speedup도 보여준다.

OPT-175B를 3bit로 양자화하면, embedding과 output layer를 FP16으로 유지하고도 model weight가 약 63GB에 들어간다. KV cache까지 고려해도 단일 80GB A100에 올릴 수 있다고 논문은 설명한다.

비교하면 다음과 같다.

| 설정 | 필요한 GPU |
|---|---:|
| FP16 OPT-175B | 5 x A100 80GB |
| LLM.int8() | 3 x A100 80GB |
| GPTQ 3bit | 1 x A100 80GB |

속도도 빨라진다. 논문은 quantized-matrix full-precision-vector product kernel을 구현해, weight를 필요할 때 dequantize하면서 matrix-vector product를 수행한다.

batch size 1 generation에서 평균 per-token latency는 다음과 같다.

| GPU | FP16 | GPTQ 3bit | Speedup | GPU 수 감소 |
|---|---:|---:|---:|---:|
| A6000 48GB | 589ms | 130ms | 4.53x | 8 -> 2 |
| A100 80GB | 230ms | 71ms | 3.24x | 5 -> 1 |

왜 속도가 빨라질까?

LLM generation은 다음 token을 하나씩 생성한다. 이때 큰 linear layer는 대체로 matrix-vector product 형태가 된다. 큰 batch의 matrix-matrix product와 달리, 이 연산은 GPU compute보다 memory bandwidth에 막히기 쉽다.

GPTQ로 weight를 3bit로 줄이면 HBM에서 읽어야 하는 weight 양이 크게 줄어든다. dequantization compute가 추가되지만, memory traffic 감소가 더 커서 latency가 줄어든다.

이 관점은 FlashAttention의 IO-aware 사고방식과도 닮아 있다.

```text
FLOPs만 줄이는 것이 아니라,
GPU 메모리에서 얼마나 읽고 쓰는지가 실제 속도를 좌우한다.
```

## 13. Group size와 extreme quantization

GPTQ는 기본 row-wise quantization만 쓸 수도 있지만, grouping과도 잘 맞는다.

![GPTQ group size]({{ '/assets/images/blog/gptq-figure4-group-size.png' | relative_url }})

*논문 Figure 4. group size를 줄이면 medium-size OPT 모델에서도 4bit GPTQ의 perplexity가 FP16에 더 가까워진다.*

grouping은 weight를 일정 크기 그룹으로 나누고, 각 그룹마다 별도의 scale/zero point를 쓰는 방식이다.

```text
큰 group:
metadata 적음
압축률 좋음
하지만 한 scale이 많은 weight를 대표해야 함

작은 group:
metadata 증가
평균 bit 수 약간 증가
하지만 quantization grid가 더 세밀하게 맞음
```

논문은 2bit 또는 ternary quantization도 실험한다. 결과가 3bit만큼 안정적인 것은 아니지만, 큰 모델에서 생각보다 버티는 모습을 보인다. 예를 들어 OPT-175B에서 2bit + group size 32 설정은 WikiText2 PPL 8.94를 기록한다. FP16 8.34와 비교하면 손실이 있지만, 2bit 수준의 압축이라는 점을 생각하면 꽤 인상적이다.

이 결과는 이후의 여러 low-bit quantization 연구가 더 공격적인 bitwidth를 실험할 수 있게 만든 기반으로 볼 수 있다.

## 14. GPTQ를 실무 관점에서 이해하기

실무적으로 GPTQ는 다음 상황에서 특히 의미가 있다.

### 큰 모델을 제한된 GPU에 올리고 싶을 때

FP16으로는 여러 GPU가 필요한 모델을 4bit 또는 3bit로 줄이면 훨씬 적은 GPU에서 실행할 수 있다. 이는 연구 환경, 개인 서버, 비용 제한이 있는 서비스에서 큰 차이를 만든다.

### batch size가 작고 generation latency가 중요할 때

논문이 노리는 speedup은 주로 low-batch autoregressive generation이다. 한 token씩 생성하는 상황에서는 matrix-vector product가 memory-bound가 되기 쉽고, weight compression의 이득이 잘 드러난다.

### retraining 없이 빠르게 압축하고 싶을 때

GPTQ는 PTQ 방식이다. fine-tuning 없이 calibration data만으로 몇 시간 안에 175B급 모델을 quantize할 수 있다. 논문 기준 OPT-175B는 약 4.2시간, BLOOM-176B는 약 3.8시간이다.

## 15. GPTQ의 한계

GPTQ가 강력하지만 만능은 아니다.

### weight-only quantization이다

논문은 activation quantization을 다루지 않는다. 즉 weight 메모리와 weight loading 비용은 줄지만, activation까지 모두 int로 계산하는 end-to-end integer inference와는 다르다.

### multiplication 자체의 compute를 줄이지 않는다

GPTQ의 속도 이득은 주로 memory movement 감소에서 온다. mainstream GPU에서 FP16 x INT4 mixed precision 연산을 직접 빠르게 지원하는 구조가 아니면, 실제 multiply compute 자체가 완전히 줄어드는 것은 아니다.

### kernel 지원이 중요하다

양자화된 weight를 저장만 해두고 매번 비효율적으로 dequantize하면 이득이 줄어든다. 논문도 별도의 GPU kernel을 구현해 speedup을 얻는다. 실제 프레임워크에서 GPTQ 성능이 좋은지는 quantization 알고리즘뿐 아니라 kernel, layout, packing 방식에 달려 있다.

### calibration data와 distribution shift 문제가 있다

GPTQ는 calibration input으로 Hessian 정보를 만든다. calibration data가 실제 서비스 입력 분포와 너무 다르면 layer reconstruction이 최적이 아닐 수 있다.

논문은 C4의 random text segment 128개만으로 좋은 결과를 보였지만, 특정 domain model이나 code model, multilingual model에서는 calibration 구성도 신경 써야 한다.

### perplexity만으로 모든 품질을 보장하지 않는다

논문도 ethics statement에서 언급하듯, perplexity나 standard accuracy가 좋아도 bias, toxicity, factuality, instruction-following 안정성 같은 secondary metric은 별도로 봐야 한다.

양자화는 모델의 내부 표현을 바꾸기 때문에, 특정 edge case에서 예상치 못한 품질 변화가 생길 수 있다.

## 16. GPTQ와 다른 양자화 방법의 관계

LLM quantization을 넓게 보면 여러 계열이 있다.

| 방법 | 핵심 아이디어 |
|---|---|
| RTN | weight를 가장 가까운 grid로 반올림 |
| LLM.int8() | 8bit quantization에서 activation outlier를 별도 처리 |
| GPTQ | Hessian 기반 weight-only PTQ |
| AWQ | 중요한 activation channel을 고려해 salient weight 보호 |
| SmoothQuant | activation outlier를 weight 쪽으로 이동해 W8A8을 쉽게 만듦 |
| GGUF/llama.cpp quant | 로컬 inference 친화적인 다양한 weight quantization format |

GPTQ의 위치는 명확하다.

```text
목표:
대형 LLM weight를 3-4bit로 줄인다.

방식:
학습 없이 calibration data로 layer-wise reconstruction을 한다.

강점:
낮은 bit에서도 RTN보다 훨씬 안정적이다.

주의:
weight-only이고, 좋은 inference kernel이 필요하다.
```

즉 GPTQ는 "양자화된 모델 포맷"이라기보다, **좋은 low-bit weight를 만들어내는 PTQ 알고리즘**으로 보는 것이 정확하다.

## 17. 내가 이해한 핵심

GPTQ의 핵심은 "rounding을 잘하자"가 아니다.

더 정확히는 다음 문제를 푸는 것이다.

```text
어떤 weight를 낮은 bit로 바꾸면 error가 생긴다.
그 error가 layer output에 미치는 영향을 보고,
아직 바꾸지 않은 weight들을 조금 조정해서 전체 output을 보존하자.
```

여기서 Hessian inverse는 "어떤 방향으로 weight를 보정해야 output error가 덜 커지는가"를 알려주는 역할을 한다.

RTN은 각 weight를 독립적으로 본다. GPTQ는 layer를 하나의 system으로 본다. 이 차이가 3bit, 4bit에서 결정적으로 벌어진다.

개인적으로 이 논문에서 가장 인상적인 부분은 알고리즘 자체보다도 scale-up 감각이다.

OBQ 같은 second-order PTQ는 이론적으로 좋아 보이지만, 거대 LLM에는 너무 느리다. GPTQ는 여기서 "정확도를 조금 포기할 수 있는 부분"과 "반드시 안정화해야 하는 부분"을 구분한다.

- greedy order는 포기한다.
- block update로 GPU utilization을 챙긴다.
- Cholesky로 수치 안정성은 지킨다.

이 조합 때문에 GPTQ는 논문 아이디어가 아니라 실제로 175B 모델을 몇 시간 안에 quantize하는 도구가 된다.

## 내가 이해한 핵심

GPTQ는 다음 문제를 해결한다.

```text
거대한 LLM을 다시 학습하지 않고,
3-4bit weight로 줄이면서,
FP16에 가까운 품질을 유지할 수 있을까?
```

논문의 답은 꽤 강하다.

- OPT-175B 4bit GPTQ는 WikiText2 PPL 8.37로 FP16 8.34와 거의 같다.
- OPT-175B 3bit GPTQ도 PPL 8.68로 유지된다.
- BLOOM-176B도 3bit GPTQ에서 PPL 8.64로 버틴다.
- RTN은 3bit에서 대부분 크게 무너진다.
- 3bit OPT-175B는 단일 A100 80GB에서 실행 가능하고, 논문 kernel 기준 generation latency도 3.24x 빨라진다.

GPTQ의 의미는 단순히 "모델 파일을 작게 만든다"가 아니다. 큰 모델을 더 적은 GPU에서, 더 낮은 latency로, retraining 없이 사용할 수 있게 만든다.

## 한 줄 요약

GPTQ는 inverse Hessian 기반 보정으로 단순 반올림의 오차 누적을 줄여, 거대 언어 모델 weight를 3-4bit로 압축해도 FP16에 가까운 perplexity를 유지하게 만든 post-training quantization 방법이다.

## 참고 자료

- Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), ICLR 2023.
- [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
