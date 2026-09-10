---
title: "Transformer는 Q·K·V를 모두 따로 만들어야 할까? K=V attention 논문 정리"
date: 2026-09-10 20:00:00 +0900
summary: "Q와 K·V만으로 attention을 구성해 KV cache를 절반으로 줄일 수 있는지 검증한 논문을 정리한다. 핵심은 Q를 독립으로 유지한 K=V가 품질 손실을 작게 유지하면서 긴 context 추론 메모리를 크게 절감했다는 점이다."
categories:
  - research
tags:
  - AI
  - LLM
  - Transformer
  - Attention
  - KV Cache
  - Inference
  - Paper
source: "Ali Kayyam et al., Do Transformers Need Three Projections? Systematic Study of QKV Variants, ICML 2026"
---

# Transformer는 Q·K·V를 모두 따로 만들어야 할까? K=V attention 논문 정리

LLM을 서빙할 때 긴 context와 높은 동시성은 곧 KV cache 메모리 문제로 이어진다. 이 글은 [Do Transformers Need Three Projections? Systematic Study of QKV Variants](https://arxiv.org/abs/2606.04032)를 바탕으로, Transformer의 Query, Key, Value 중 Key와 Value를 하나로 묶어도 되는지 정리한다.

논문의 결론을 먼저 말하면 다음과 같다.

> **Q는 독립으로 남기고 K와 V만 공유하는 `Q ≠ K = V` 구조는, 언어 모델 품질을 조금만 잃으면서 KV cache를 절반으로 줄일 수 있다.**

이는 attention을 없애는 방법이 아니다. attention이 정보를 찾고 가져오는 방식은 유지하되, Key와 Value에 별도 projection이 반드시 필요한지 검증한 연구다.

## 1. 배경: Q, K, V는 각각 왜 필요한가

표준 self-attention에서 입력 표현 `X`는 세 개의 선형 projection을 거친다.

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V
$$

그리고 attention output은 다음과 같다.

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad \mathrm{output}=AV
$$

- **Q(Query)**: 현재 토큰이 무엇을 찾는지 나타낸다.
- **K(Key)**: 각 토큰이 어떤 특징을 가진 위치인지 나타낸다.
- **V(Value)**: attention이 선택해서 다음 층으로 전달할 내용을 나타낸다.

autoregressive decoding에서는 과거 토큰의 `K`, `V`를 매번 다시 만들지 않기 위해 KV cache에 저장한다. 따라서 표준 구조의 cache는 대략 다음에 비례한다.

$$
\mathrm{KV\ cache} \propto 2 \times B \times S \times L \times H_{kv} \times D
$$

여기서 앞의 `2`는 Key와 Value를 모두 저장한다는 뜻이다. 긴 prompt를 다루거나 많은 사용자를 동시에 처리할수록 이 두 tensor를 보관하고 읽는 비용이 커진다.

## 2. 논문의 질문: K와 V는 정말 분리되어야 할까

저자들은 QKV의 세 projection 중 일부를 같은 가중치로 묶는 세 구조를 비교했다.

| 구조 | attention 식 | 직관 |
| --- | --- | --- |
| 표준 QKV | `softmax(QKᵀ)V` | 검색용 K와 전달용 V가 분리됨 |
| `Q = K ≠ V` | `softmax(KKᵀ)V` | Q와 K를 공유 |
| `Q ≠ K = V` | `softmax(QKᵀ)K` | K와 V를 공유 |
| `Q = K = V` | `softmax(KKᵀ)K` | 세 역할 모두 공유 |

여기서 가장 중요한 후보는 `Q ≠ K = V`다.

```text
기존: x -> Wq -> q
          -> Wk -> k
          -> Wv -> v

K=V: x -> Wq  -> q
         -> Wkv -> kv (= k = v)
```

`K=V`에서는 Key가 동시에 두 역할을 한다.

1. Q와의 내적으로 어느 토큰을 참고할지 결정한다.
2. attention weight로 가중합되어 실제 문맥 정보를 전달한다.

따라서 마지막 가중합이 사라지는 것은 아니다.

$$
\mathrm{output}=\underbrace{\mathrm{softmax}(QK^\top)}_{\text{어디를 볼지}}\underbrace{K}_{\text{가져올 정보}}
$$

없어지는 것은 `V=XW_V`라는 **별도의 Value projection**과 별도의 V cache다.

## 3. 학습과 추론은 어떻게 달라지는가

### 학습: 별도 V projection을 만들지 않는다

학습 코드는 개념적으로 다음처럼 바뀐다.

```python
# 표준 QKV
q = x @ Wq
k = x @ Wk
v = x @ Wv
output = softmax(q @ k.transpose(-1, -2)) @ v

# Q != K = V
q = x @ Wq
kv = x @ Wkv
output = softmax(q @ kv.transpose(-1, -2)) @ kv
```

loss, causal mask, residual connection, MLP, optimizer 같은 나머지 Transformer 학습 절차는 바뀌지 않는다. 다만 `Wkv`는 주소 역할과 내용 역할 양쪽에서 gradient를 받으므로, 두 역할을 동시에 잘 수행하도록 처음부터 학습된다.

이미 학습된 QKV 모델의 `Wk`, `Wv`를 단순히 평균 내어 바꾸는 것은 이 논문이 검증한 방법이 아니다. 논문의 핵심 결과는 이 제약을 둔 모델을 **from scratch로 학습**했을 때 나온 결과다. 실제 도입에서는 새로 pretraining하거나 충분한 uptraining/fine-tuning 검증이 필요하다.

### 추론: K/V 두 cache 대신 하나만 저장한다

decode 단계에서 표준 Transformer는 새 토큰마다 `K`와 `V`를 cache에 append한다.

```text
표준 QKV
K_cache = [K1, K2, ..., Kt]
V_cache = [V1, V2, ..., Vt]

K=V
KV_cache = [KV1, KV2, ..., KVt]
```

새 토큰의 Q로 `KV_cache` 전체에 attention score를 계산하고, 같은 cache를 가중합에 다시 쓴다. 그래서 cache 용량과 decode 중 cache를 읽는 메모리 트래픽이 50% 줄어든다.

```python
q_new = x_new @ Wq
kv_new = x_new @ Wkv
kv_cache.append(kv_new)

weights = softmax(q_new @ kv_cache.transpose(-1, -2))
output = weights @ kv_cache
```

## 4. 비용은 얼마나 줄어드는가

절감 효과는 학습과 추론에서 성격이 다르다.

| 항목 | `Q ≠ K = V`의 변화 | 이유 |
| --- | --- | --- |
| attention projection | 약 33% 감소 | Q/K/V 3회 projection이 Q/KV 2회가 됨 |
| attention projection 파라미터 | 약 33% 감소 | 세 행렬 중 한 개를 제거 |
| 전체 모델 파라미터 | 300M 설정에서 약 6.9% 감소 | MLP와 embedding은 그대로임 |
| 전체 추론 MAC | 길이 2,048에서 약 5.4% 감소 | attention 외 MLP·LM head 비용이 큼 |
| KV cache | **50% 감소** | K와 V를 한 tensor로 저장 |

즉, 학습 FLOPs가 획기적으로 반으로 줄어드는 구조는 아니다. 큰 MLP가 전체 연산과 파라미터에서 차지하는 비중이 크기 때문이다.

반대로 긴 context의 decode에서는 KV cache가 메모리 용량과 memory bandwidth의 병목이 되기 쉽다. 이 경우 cache를 절반으로 줄이는 효과가 매우 실용적이다. 같은 GPU 메모리에서 더 긴 context를 쓰거나, 더 많은 요청을 동시에 처리할 여지가 생긴다.

## 5. 실험 결과: K=V는 얼마나 버티는가

저자들은 합성 추론 과제, vision 과제, SlimPajama 기반 언어 모델 pretraining을 비교했다. 언어 모델은 300M과 1.2B parameter 모델을 최대 10B token으로 학습했다.

### 300M 언어 모델

Validation perplexity(PPL)는 낮을수록 좋다.

| 방식 | Validation PPL | QKV 대비 PPL 변화 | KV cache 절감 |
| --- | ---: | ---: | ---: |
| QKV | 5.11 | 기준 | - |
| `Q ≠ K = V` | 5.27 | +3.1% | **50%** |
| `Q = K ≠ V` | 5.36 | +4.9% | 0% |
| `Q = K = V` | 6.41 | +25.4% | 50% |

`K=V`는 PPL이 3.1% 나빠졌지만 QKV에 가깝게 학습 곡선을 따라갔다. 반면 Q와 K를 공유하는 구조는 cache 이득 없이 더 나쁜 결과를 냈고, 세 projection을 모두 공유한 구조는 언어 모델에서 너무 큰 제약이었다.

### 1.2B 언어 모델과 downstream 평가

더 큰 1.2B 모델에서는 `Q ≠ K = V`의 PPL이 5.004에서 5.128로 변했다. QKV 대비 **2.48% 악화**이면서 KV cache는 여전히 절반이었다.

5-shot downstream benchmark(ARC, HellaSwag, PIQA, WinoGrande) 평균 정확도도 QKV 36.40%에서 `Q ≠ K = V` 35.99%로, **0.41%p 차이**였다. 논문 결과만 보면 perplexity의 작은 차이가 downstream capability의 같은 크기 손실로 직결되지는 않았다.

## 6. 왜 K=V는 되고 Q=K는 약한가

저자들의 해석은 간단하다.

- K와 V는 이미 꽤 비슷한 표현 공간을 쓸 수 있다. 학습된 QKV 모델에서 K/V projection의 layer별 cosine similarity는 0.73으로 보고됐다.
- Q는 K/V와 더 다르다. Q-K similarity는 0.42, Q-V similarity는 0.31이었다.
- Q는 “현재 위치에서 무엇을 찾을지”, K/V는 “과거 위치가 무엇을 나타내는지”라는 서로 다른 역할을 갖는다.

`Q=K`에서는 score matrix가 `KKᵀ`가 되어 대칭적인 성질을 강하게 띤다. 이는 “A가 B를 참고하는 관계”와 “B가 A를 참고하는 관계”를 다르게 만들기 어려운 방향으로 작용한다. causal language modeling처럼 순서와 방향성이 중요한 문제에는 불리할 수 있다.

반면 `K=V`는 `QKᵀ`의 비대칭성을 보존한다. 모델은 여전히 Q로 어떤 과거 위치를 읽을지 동적으로 선택할 수 있고, 선택된 K를 내용으로도 활용한다.

## 7. GQA/MQA와 함께 쓰면

K=V는 GQA/MQA와 경쟁하는 기법이 아니라, 서로 다른 축의 최적화다.

- **GQA/MQA**: 여러 Query head가 더 적은 수의 KV head를 공유한다.
- **K=V**: 각 KV head에서 K와 V tensor 자체를 하나로 합친다.

그래서 함께 적용할 수 있다. 논문의 1.2B 결과는 다음과 같다.

| 방식 | QKV 대비 PPL 변화 | KV cache 절감 |
| --- | ---: | ---: |
| GQA-8 | +0.52% | 75% |
| MQA | +1.06% | 93.8% |
| K=V + GQA-8 | +3.08% | 87.5% |
| K=V + MQA | +4.16% | 96.9% |

cache를 아주 공격적으로 줄여야 한다면 결합 방식이 매력적이다. 다만 이 결과만 놓고 품질 우선의 운영 모델에 K=V를 무조건 추가할 이유는 없다. 논문에서도 GQA/MQA 단독이 더 좋은 품질-메모리 trade-off를 보이는 지점이 있다.

## 8. 한계: 아직 표준 QKV를 대체했다고 보기는 이르다

이 논문은 흥미로운 controlled study지만, 다음 범위를 벗어난 결론은 조심해야 한다.

- 가장 큰 검증 모델은 **1.2B parameter**다. 7B, 70B 이상에서도 손실이 작게 유지되는지는 확인되지 않았다.
- 학습량은 최대 **10B tokens**다. 대형 LLM의 일반적 학습 규모와는 차이가 있다.
- 실험 sequence length는 최대 **2,048 tokens**다. 장점이 특히 큰 32K/128K 장문 context에서 품질과 실제 latency를 충분히 검증하지 않았다.
- K/V가 유사하다는 설명은 관찰에 기반한다. 모든 아키텍처·데이터·학습 방식에 통하는 이론적 보장은 없다.
- code generation, instruction tuning, RLHF, 멀티모달, encoder-decoder cross-attention처럼 실전 모델에서 중요한 조건을 폭넓게 다루지는 않았다.
- cache가 줄어든다고 latency가 언제나 같은 비율로 줄어들지는 않는다. attention score 계산과 MLP 연산은 여전히 남아 있고, serving engine·batching·hardware에 따라 병목이 달라진다.

## 9. 정리

이 논문의 메시지는 “attention이 필요 없다”가 아니다. **K와 V의 분리는 생각보다 중복일 수 있다**는 것이다.

```text
Q는 유지한다: 현재 토큰이 무엇을 읽을지 결정한다.
K와 V는 합친다: 위치를 나타내는 표현이 전달할 내용도 겸할 수 있다.
```

따라서 `Q ≠ K = V`는 특히 다음 조건에서 검토할 만하다.

- 긴 context 때문에 KV cache가 큰 문제인 경우
- GPU 메모리보다 추론 품질의 미세한 차이를 일부 더 양보할 수 있는 경우
- edge 또는 on-device처럼 메모리 제약이 강한 경우
- 기존 GQA/MQA 위에 추가 cache 절감을 실험하려는 경우

반면 이미 품질-효율 균형이 검증된 GQA/MQA 기반 대형 모델을 운영 중이라면, K=V는 즉시 적용할 기본값이라기보다 **재학습과 장문·대규모 benchmark 검증이 필요한 유망한 설계 후보**로 보는 편이 적절하다.

## 참고 자료

- Ali Kayyam, Anusha Madan Gopal, M. Anthony Lewis. [Do Transformers Need Three Projections? Systematic Study of QKV Variants](https://arxiv.org/abs/2606.04032), ICML 2026.
- [GQA(Grouped-Query Attention) 이해하기: MHA와 MQA 사이의 균형점](/2026/06/23/gqa/)
- [Long Context에서 KV Cache가 병목이 되는 이유](/2026/06/21/large-kv-cache-long-context/)
