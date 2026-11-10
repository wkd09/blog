---
title: "Mamba 1 논문 정리: 선택적 상태 공간 모델로 Attention 없이 긴 문맥 처리하기"
date: 2026-08-31 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
summary: "Mamba가 기존 State Space Model에 입력 의존적 selection을 도입한 이유, selective scan의 GPU 최적화, Mamba block의 구조와 언어 모델 실험 결과를 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - Mamba
  - State Space Model
  - SSM
  - Selective Scan
  - Sequence Modeling
  - Paper
source: "arXiv:2312.00752"
---

# Mamba 1: 필요한 정보만 State에 남기는 Sequence Model

이 글은 논문 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)를 바탕으로 Mamba 1의 핵심 아이디어를 정리한 글이다.

> Albert Gu, Tri Dao  
> [[Paper](https://arxiv.org/pdf/2312.00752)] [[Code](https://github.com/state-spaces/mamba)]

Transformer의 self-attention은 현재 token이 과거의 어느 위치를 참고할지 직접 선택한다. 이 방식은 언어 모델에서 강력하지만 sequence length가 길어질수록 attention 계산량이 제곱으로 증가한다. Autoregressive inference에서는 과거 전체의 KV cache도 보관해야 한다.

State Space Model(SSM)은 과거를 고정된 크기의 hidden state로 압축한다. Sequence length에 선형으로 계산할 수 있고, 생성할 때도 길이에 비례하는 KV cache가 필요하지 않다.

문제는 기존 SSM의 state transition이 입력 내용과 관계없이 고정되어 있었다는 점이다. 어떤 token이 중요한지 보고 기억할지 버릴지 결정하기 어려웠다. 이 한계는 audio 같은 연속 신호보다 정보가 빽빽한 text에서 더 크게 나타났다.

Mamba는 이 문제를 다음 세 단계로 해결한다.

1. SSM의 $\Delta$, $B$, $C$를 현재 입력의 함수로 만들어 **필요한 정보만 기억한다.**
2. 이 변화로 convolution을 사용할 수 없게 된 문제를 **hardware-aware selective scan**으로 해결한다.
3. Selective SSM과 gated MLP를 하나의 단순한 **Mamba block**으로 합친다.

이 글에서는 기존 SSM이 왜 text에서 약했는지부터 시작해 selection mechanism, selective scan, Mamba block과 실험 결과를 순서대로 정리한다.

핵심 관심사는 하나다.

> 긴 문맥 전체를 저장하지 않고도, 필요한 정보를 골라 고정 크기 state에 남길 수 있는가?

논문에서 직접 확인되는 구조와 실험 결과를 먼저 설명하고, 이해를 돕기 위한 해석은 따로 구분해 적었다.

## 1. Attention과 Recurrent Model의 차이

논문은 sequence modeling의 핵심을 **context compression**으로 바라본다.

Attention은 과거 token을 거의 압축하지 않는다. 각 token의 Key와 Value를 KV cache에 남겨두고, 새로운 query가 들어올 때 과거 전체에서 필요한 위치를 다시 찾는다.

```text
Attention
[x1][x2][x3] ... [xt]
 └──── 과거 token의 KV를 모두 보관 ────┘
                         query로 필요한 위치 선택
```

반면 RNN과 SSM은 지금까지의 문맥을 하나의 고정된 state $h_t$에 누적한다.

```text
Recurrent model
h(t-1) + x(t) -> h(t) -> y(t)
                  └ 과거를 고정 크기로 압축
```

두 방식의 차이는 결국 과거를 얼마나 압축하는가에 있다.

| 모델 | 과거를 저장하는 방식 | 장점 | 핵심 부담 |
| --- | --- | --- | --- |
| Self-Attention | token별 KV를 보존 | 필요할 때 과거 위치를 직접 조회 | 학습 attention 계산 $O(L^2)$, KV cache $O(L)$ |
| RNN / SSM | 고정 크기 state로 압축 | 학습 $O(L)$, 생성 step당 state 크기 $O(1)$ | 압축 과정에서 필요한 정보가 사라질 수 있음 |

여기서 $O(1)$은 sequence length $L$에 대한 표현이다. 실제 비용은 layer 수, model dimension, state dimension에 당연히 의존한다.

Attention은 과거를 저장해두고 나중에 필요한 위치를 선택한다. Mamba는 **입력이 들어오는 순간 무엇을 state에 남길지 선택**한다.

즉 Mamba의 핵심은 고정 크기 state 자체가 아니다. 제한된 state에 필요한 정보를 얼마나 잘 압축하는지가 중요하다.

## 2. 기존 SSM은 어떻게 동작하는가

### Continuous State Space Model

State Space Model은 연속 시간의 dynamic system을 표현하는 방식에서 출발한다. 입력 $x(t)$를 latent state $h(t)$에 반영하고, state에서 출력 $y(t)$를 읽는다.

$$
h'(t) = Ah(t) + Bx(t)
$$

$$
y(t) = Ch(t)
$$

각 parameter는 다음처럼 볼 수 있다.

- $A$: 이전 state가 시간에 따라 어떻게 변화하고 유지되는가
- $B$: 현재 입력을 state에 어떻게 기록하는가
- $C$: state의 어느 정보를 출력으로 읽는가
- $\Delta$: 연속 시스템을 discrete token 간격으로 바꾸는 step size

### Discretization

Text는 연속 신호가 아니라 token sequence다. 따라서 위 식을 discrete recurrence로 바꿔야 한다. Zero-order hold를 사용하면 continuous parameter $A$, $B$는 $\Delta$를 통해 $\bar A$, $\bar B$로 변환된다.

$$
\bar A = \exp(\Delta A)
$$

$$
\bar B = (\Delta A)^{-1}\big(\exp(\Delta A)-I\big)\Delta B
$$

이후 실제 sequence는 다음 recurrence로 처리한다.

$$
h_t = \bar A h_{t-1} + \bar B x_t
$$

$$
y_t = Ch_t
$$

이 식은 RNN처럼 왼쪽에서 오른쪽으로 계산할 수 있다. 동시에 $A$, $B$, $C$, $\Delta$가 모든 timestep에서 고정된 **Linear Time-Invariant(LTI)** system이라면 하나의 convolution kernel로도 바꿀 수 있다.

$$
K = \left(C\bar B,\; C\bar A\bar B,\; C\bar A^2\bar B,\; \dots\right)
$$

$$
y = x * K
$$

그래서 기존 structured SSM은 두 가지 계산 mode를 사용할 수 있었다.

```text
Training  : 전체 sequence를 알고 있음 -> global convolution으로 병렬 계산
Inference : token이 하나씩 들어옴      -> recurrence로 state 갱신
```

같은 SSM을 training에서는 convolution으로, inference에서는 recurrence로 계산할 수 있다는 것이 장점이다.

하지만 이 장점을 사용하려면 parameter가 시간에 따라 변하지 않아야 한다. 이 제약이 기존 SSM의 약점으로 이어진다.

## 3. 기존 SSM은 왜 Text에서 약했는가

고정된 transition은 “몇 칸 뒤에 무엇을 출력하라” 같은 시간 기반 pattern은 잘 처리한다. 그러나 실제 text에서 중요한 것은 위치만이 아니라 **내용**이다.

논문은 이 차이를 Copying과 Selective Copying task로 설명한다.

![Copying, Selective Copying과 Induction Heads task]({{ '/assets/images/blog/mamba-selective-copying-induction.png' | relative_url }})

*일반 Copying은 간격이 고정되어 있어 정적인 convolution kernel도 풀 수 있다. Selective Copying은 유효한 token의 위치가 매번 달라 입력 내용을 보고 기억할지 결정해야 한다. Induction Heads는 앞에서 본 pattern을 현재 context에 맞춰 회상해야 한다. 출처: Mamba Figure 2.*

일반 Copying task에서는 복사할 token과 출력 위치 사이의 간격이 고정되어 있다. 모델은 token의 의미를 보지 않고 정확한 시차만 학습해도 문제를 풀 수 있다.

Selective Copying에서는 복사할 colored token 사이에 길이가 제각각인 noise가 들어간다.

```text
고정 Copying
[A][B][C] ---- 고정 간격 ----> [A][B][C]

Selective Copying
[noise][A][noise][noise][B][noise][C] -> [A][B][C]
        기억              기억       기억
```

Static convolution은 모든 위치에 같은 kernel을 적용한다. 현재 token이 중요한 값인지 noise인지에 따라 state update rule을 바꿀 수 없다. Induction Heads처럼 앞에서 등장한 key-value pattern을 context에 따라 찾아야 하는 associative recall에서도 같은 문제가 나타난다.

즉 고정 크기 state 자체가 문제라기보다 **무엇을 state에 넣고 뺄지 입력에 맞춰 결정하지 못하는 것**이 문제다.

## 4. 핵심 아이디어: 입력에 따라 State를 선택한다

Mamba의 selective SSM은 논문에서 S6라고 부른다. S6는 $A$를 제외한 $\Delta$, $B$, $C$를 현재 입력 $x_t$의 함수로 만든다.

$$
B_t = s_B(x_t), \qquad C_t = s_C(x_t)
$$

$$
\Delta_t = \operatorname{softplus}\left(\Delta_{\text{parameter}} + s_\Delta(x_t)\right)
$$

Discretization을 거치면 recurrence도 token마다 달라진다.

$$
h_t = \bar A_t h_{t-1} + \bar B_t x_t
$$

$$
y_t = C_t h_t
$$

![Mamba의 Selective State Space Model]({{ '/assets/images/blog/mamba-selective-ssm-overview.png' | relative_url }})

*입력 $x_t$에서 $B_t$, $C_t$, $\Delta_t$를 만들고, $\Delta_t$와 고정된 $A$를 discretize해 현재 token의 state transition을 결정한다. 확장된 state는 GPU HBM에 저장하지 않고 빠른 SRAM 안에서 계산한다. 출처: Mamba Figure 1.*

이 세 parameter는 다음처럼 이해할 수 있다.

| Parameter | 선택의 의미 | 직관 |
| --- | --- | --- |
| $\Delta_t$ | 이전 state를 얼마나 유지하거나 갱신할지 결정 | forget gate / reset 강도 |
| $B_t$ | 현재 token의 어떤 정보를 state에 기록할지 결정 | selective write |
| $C_t$ | 현재 state의 어떤 정보를 출력할지 결정 | selective read |

위 표는 이해를 위한 해석이다. 실제 계산에서는 $\Delta_t$가 discretization을 통해 $\bar A_t$, $\bar B_t$를 함께 바꾼다.

### $\Delta$는 왜 Gate처럼 동작하는가

논문은 $N=1$, $A=-1$, $B=1$인 단순한 조건에서 selective SSM이 다음 식으로 정리된다는 것을 보인다.

$$
g_t = \sigma(\operatorname{Linear}(x_t))
$$

$$
h_t = (1-g_t)h_{t-1} + g_tx_t
$$

$g_t$가 0에 가까우면 현재 입력을 무시하고 이전 state를 보존한다. 1에 가까우면 이전 state를 지우고 현재 입력으로 갱신한다. LSTM과 GRU의 gate가 하던 역할을 SSM의 discretization 관점에서 얻는 셈이다.

```text
g(t) ≈ 0 -> h(t) ≈ h(t-1)  -> 현재 token 무시
g(t) ≈ 1 -> h(t) ≈ x(t)    -> 현재 token을 쓰고 state reset
```

이 selection mechanism은 세 가지 효과를 만든다.

- **Variable spacing**: 중요한 token 사이에 noise가 얼마나 끼어 있어도 무시할 수 있다.
- **Context filtering**: 긴 문맥에서 불필요해진 과거를 지우고 필요한 정보만 남길 수 있다.
- **Boundary resetting**: 여러 document를 packing했을 때 경계에서 state를 초기화해 정보가 섞이는 것을 막을 수 있다.

## 5. Selection을 넣으면 Convolution을 사용할 수 없다

이제 $B_t$, $C_t$, $\Delta_t$가 token마다 달라진다. 따라서 하나의 고정된 convolution kernel $K$를 미리 만들 수 없다.

```text
기존 SSM
같은 A, B, C, Δ -> 고정 kernel K -> convolution 가능

Selective SSM
token마다 B(t), C(t), Δ(t) -> kernel도 계속 변함 -> 고정 convolution 불가능
```

내용에 따라 state를 바꾸는 능력을 얻었지만, 이제 recurrence를 직접 계산해야 한다. Naive implementation은 두 가지 이유로 느리다.

1. $h_t$가 $h_{t-1}$에 의존하므로 순차적으로 보인다.
2. Batch $B$, length $L$, channel $D$, state dimension $N$에 대해 확장 state의 크기가 $O(BLDN)$이 된다.

특히 $(B,L,D,N)$ 크기의 $\bar A$, $\bar B$와 중간 state를 GPU HBM에 쓰고 다시 읽으면 계산보다 memory I/O가 병목이 된다.

Mamba의 두 번째 핵심은 이 recurrence를 GPU에서 실제로 빠르게 실행한 것이다.

## 6. Selective Scan을 GPU에서 빠르게 계산하는 방법

### Parallel Scan

Recurrence는 순차적으로 보이지만 각 step을 associative operator로 표현하면 prefix scan으로 병렬화할 수 있다. 단순화해서 한 step을 다음 pair로 보자.

$$
h_t = a_t h_{t-1} + b_t
$$

두 연속 step $(a_1,b_1)$, $(a_2,b_2)$를 합치면 다음과 같다.

$$
(a_2,b_2) \circ (a_1,b_1)
= \left(a_2a_1,\; a_2b_1+b_2\right)
$$

이 합성은 associative하므로 tree 형태의 parallel scan을 적용할 수 있다. Selection 때문에 고정 convolution은 사용할 수 없지만 scan 자체는 병렬화할 수 있다.

### Kernel Fusion과 GPU Memory Hierarchy

실제 구현에서 더 중요한 것은 FLOPs보다 memory movement다.

```text
Naive
HBM에서 parameter 읽기
-> Δ, A, B discretize
-> (B,L,D,N) 중간 tensor를 HBM에 기록
-> 다시 읽어 scan
-> output 기록

Mamba selective scan
HBM에서 Δ, A, B, C 읽기
-> SRAM에서 discretize + scan을 fused kernel로 수행
-> (B,L,D) output만 HBM에 기록
```

Mamba는 확장된 state를 크고 상대적으로 느린 HBM에 materialize하지 않는다. Parameter를 HBM에서 빠른 on-chip SRAM으로 불러온 뒤 discretization과 scan을 하나의 fused kernel 안에서 처리한다. HBM에는 최종 output만 쓴다.

Backward pass에 필요한 모든 중간 state도 저장하지 않는다. 대신 backward 때 HBM에서 입력을 다시 읽어 SRAM 안에서 state를 recompute한다. 계산은 조금 늘지만 훨씬 비싼 HBM I/O와 memory footprint를 줄인다. 논문은 이 방식의 memory 요구량이 FlashAttention을 사용한 최적화 Transformer와 비슷하다고 설명한다.

정리하면 selective scan의 핵심은 다음과 같다.

> Mamba는 recurrence를 없앤 것이 아니라, recurrence의 중간 결과가 GPU memory hierarchy의 느린 구간을 왕복하지 않도록 계산 경로를 다시 설계했다.

## 7. Mamba Block 구조

Selective SSM은 하나의 sequence transformation layer다. 논문은 기존 H3 block과 gated MLP의 구조를 합쳐 Mamba block을 만든다.

![H3, Gated MLP와 Mamba block 비교]({{ '/assets/images/blog/mamba-block-architecture.png' | relative_url }})

*Mamba는 gated MLP의 main branch에 Conv와 selective SSM을 넣은 형태로 볼 수 있다. H3 block과 별도의 MLP block을 번갈아 쌓지 않고 같은 Mamba block을 반복한다. 출처: Mamba Figure 3.*

한 block의 흐름을 단순화하면 다음과 같다.

```text
                           ┌-> Linear -> SiLU ----------------┐
input -> normalization ----┤                                 × -> Linear -> residual
                           └-> Linear -> Conv1D -> SiLU -> SSM┘
```

Main branch의 local convolution은 가까운 token 사이의 pattern을 먼저 처리하고, selective SSM은 긴 sequence 방향의 정보를 처리한다. 다른 branch는 SiLU를 통과한 gate가 되어 main branch의 출력을 조절한다. 마지막으로 output projection을 거쳐 residual connection에 더한다.

Transformer block과 비교하면 다음 차이가 있다.

| Transformer block | Mamba block |
| --- | --- |
| Attention block과 MLP block이 분리 | SSM과 gate를 하나의 block에 결합 |
| 위치 간 interaction을 attention matrix로 계산 | recurrent state를 selective scan으로 갱신 |
| Autoregressive inference에 KV cache 필요 | Layer별 고정 크기 recurrent state 필요 |
| Attention 계산은 length에 대해 quadratic | Scan 계산은 length에 대해 linear |

Mamba architecture에는 attention layer와 별도의 MLP block이 없다. 다만 linear projection, local convolution, activation, normalization, residual connection은 사용한다.

즉 attention-free라는 말이 행렬 곱셈까지 없는 모델이라는 뜻은 아니다.

## 8. 복잡도와 Inference 특성

Sequence length를 $L$이라 하면 핵심 차이는 다음과 같다.

| 항목 | Transformer | Mamba |
| --- | --- | --- |
| 전체 sequence의 핵심 mixing 계산 | Self-attention $O(L^2)$ | Selective scan $O(L)$ |
| Autoregressive step에서 과거 접근 | 과거 $L$개 KV 조회 | 현재 recurrent state만 조회 |
| Sequence 길이에 따른 inference state | $O(L)$ KV cache | $O(1)$ recurrent state |
| 긴 sequence training | Attention FLOPs 증가 | Length에 선형 증가 |

Mamba가 생성 step마다 유지하는 state 크기는 sequence length와 무관하다. 따라서 context가 길어져도 KV cache처럼 memory가 계속 자라지 않는다. 한 token을 추가하는 recurrence 비용도 과거 길이에 따라 증가하지 않는다.

문제는 한 번 압축한 정보를 다시 꺼낼 수 없다는 점이다. Transformer는 필요할 때 과거 token의 KV를 직접 읽을 수 있지만 Mamba는 압축된 state만 가진다. 과거의 세부 정보가 state에서 사라졌다면 원본 token으로 돌아가 복구할 수 없다.

Selectivity가 중요한 이유가 여기에 있다.

## 9. 실험 결과 요약

### Selective Copying과 Induction Heads

논문 기준으로 Selective Copying ablation에서 Mamba architecture에 기존 S4 또는 Hyena layer를 넣으면 정확도가 각각 56.4%, 28.4%였다. Selective SSM인 S6를 넣은 Mamba는 99.8%를 기록했다.

Induction Heads task에서는 길이 256으로 학습한 Mamba가 1,048,576 길이까지 정확하게 extrapolation했다. 논문에서 다른 비교 방법은 학습 길이의 2배를 넘어서면서 성능이 무너졌다.

이 결과는 block 구조보다 입력에 따라 state를 선택하는 mechanism이 중요하다는 것을 보여준다.

### Language Modeling Scaling Law

![The Pile에서 Mamba의 scaling law]({{ '/assets/images/blog/mamba-scaling-laws.png' | relative_url }})

*The Pile에서 약 125M부터 1.3B parameter 모델을 비교한 scaling law. Sequence length 2048과 8192 모두에서 Mamba가 다른 attention-free model보다 낮은 perplexity를 보였고 강한 Transformer++ recipe와 경쟁했다. 출처: Mamba Figure 4.*

논문은 GPT-3 규모 설정을 따라 약 125M부터 1.3B까지 모델을 키우며 The Pile에서 비교했다. Transformer++는 RoPE, SwiGLU, RMSNorm, bias 제거, 높은 learning rate 등 PaLM과 LLaMA 계열의 강한 recipe를 적용한 baseline이다.

Mamba는 Hyena, RWKV, RetNet, H3++ 같은 subquadratic architecture보다 좋은 scaling curve를 보였다. 특히 sequence length 8192에서 강한 Transformer++와 비슷하거나 더 좋은 perplexity를 기록했다.

즉 기존 linear-time model이 language modeling에서 Transformer 품질을 따라가지 못하던 격차를 줄였다.

### Zero-shot Evaluation

![Mamba의 language modeling과 zero-shot 결과]({{ '/assets/images/blog/mamba-language-model-results.png' | relative_url }})

*LAMBADA perplexity와 accuracy, HellaSwag, PIQA, ARC, WinoGrande의 zero-shot 결과. 굵은 글씨는 각 규모 구간의 Mamba 성능이다. 출처: Mamba Table 3.*

대표 결과를 추리면 다음과 같다.

| Model | The Pile PPL ↓ | LAMBADA PPL ↓ | Zero-shot 평균 ↑ |
| --- | ---: | ---: | ---: |
| Pythia-1.4B | 7.51 | 6.08 | 55.2 |
| Mamba-1.4B | **6.80** | **5.04** | **59.7** |
| Pythia-2.8B | 6.73 | 5.04 | 59.1 |
| RWKV-3B | 7.00 | 5.24 | 59.6 |
| Mamba-2.8B | **6.22** | **4.23** | **63.3** |
| Pythia-6.9B | 6.51 | 4.45 | 61.7 |

Mamba-2.8B의 zero-shot 평균 63.3은 Pythia-2.8B의 59.1뿐 아니라 Pythia-6.9B의 61.7보다 높았다. 저자들이 Mamba-3B가 약 두 배 크기의 Transformer와 경쟁한다고 설명한 근거다.

다만 모든 모델이 같은 tokenizer와 architecture를 사용한 것은 아니다. 표에도 GPT-2, NeoX, OPT tokenizer가 섞여 있다. 따라서 이 수치는 논문이 사용한 baseline과 training recipe 안에서 봐야 한다.

### 속도와 Memory

![Mamba selective scan과 inference throughput benchmark]({{ '/assets/images/blog/mamba-efficiency-benchmarks.png' | relative_url }})

*왼쪽은 A100 80GB PCIe에서 sequence length에 따른 scan, convolution, FlashAttention-2 시간이다. 오른쪽은 prompt length 2048에서 batch size별 생성 throughput이다. 출처: Mamba Figure 8.*

논문의 A100 80GB PCIe benchmark에서 fused selective scan은 sequence length 2K 이후 FlashAttention-2보다 빨랐고, 표준 PyTorch scan보다 20~40배 빨랐다.

Prompt length 2048의 autoregressive inference에서는 동급 크기 Transformer보다 약 4~5배 높은 throughput을 기록했다. Mamba-1.4B는 batch size 128에서 1,814 tokens/s까지 증가한 반면 Transformer-1.3B는 batch size 32 이후 KV cache memory 때문에 OOM이 발생했다.

여기서 중요한 것은 single-request latency보다 batch를 키웠을 때의 throughput이다. Mamba는 KV cache가 없어 같은 GPU memory에 더 큰 batch를 넣을 수 있다.

반대로 작은 batch, 짧은 sequence, 다른 GPU와 kernel 환경에서 항상 4~5배 빨라진다는 뜻은 아니다.

### Language를 넘어선 결과

논문은 Mamba를 범용 sequence backbone으로 제안하며 DNA와 raw audio도 평가했다.

- Great Apes DNA classification에서 sequence length가 최대 약 100만까지 길어질수록 성능이 개선되었다.
- YouTubeMix raw audio modeling에서도 약 100만 sample, 즉 1분가량의 context까지 길이가 늘면서 성능이 좋아졌다.
- Speech generation에서는 Mamba block으로 구성한 model이 비교한 S4와 attention 조합보다 가장 낮은 FID를 기록했다.

이는 selective SSM이 text 전용 trick이 아니라 긴 sequence를 고정 크기 state로 압축하는 일반 mechanism일 가능성을 보여준다.

## 10. Ablation에서 확인한 핵심

논문의 ablation을 보면 Mamba의 성능이 어디에서 나오는지 더 분명해진다.

### Architecture보다 Selectivity가 중요하다

약 350M language model에서 inner sequence layer만 바꿨을 때 perplexity는 다음과 같았다.

| Architecture | Inner layer | Perplexity ↓ |
| --- | --- | ---: |
| H3 | S4 (complex) | 10.30 |
| H3 | S6 | 8.95 |
| Mamba | S4 (real) | 10.56 |
| Mamba | S6 | **8.69** |

Mamba block에 non-selective S4를 넣는 것만으로는 성능이 좋아지지 않았다. 반대로 H3 architecture라도 selective S6를 넣으면 크게 개선됐다.

즉 단순한 block 재배치보다 selection mechanism이 핵심이다.

### $\Delta$, $B$, $C$를 함께 선택할 때 가장 좋다

| Selective $\Delta$ | Selective $B$ | Selective $C$ | Perplexity ↓ |
| --- | --- | --- | ---: |
| X | X | X | 10.93 |
| X | O | X | 10.15 |
| X | X | O | 9.98 |
| O | X | X | 9.81 |
| O | O | O | **8.71** |

$\Delta$ 하나만 selective하게 만들어도 가장 큰 단일 개선이 생긴다. 하지만 $\Delta$, $B$, $C$를 모두 입력 의존적으로 만들었을 때 성능이 가장 좋았다.

즉 “언제 갱신할지”, “무엇을 쓸지”, “무엇을 읽을지”가 함께 움직여야 한다.

또한 $B$, $C$가 selective할 때 state dimension $N$을 1에서 16으로 키우면 perplexity가 9.73에서 8.71로 개선됐다. 반대로 $B$, $C$가 고정이면 같은 확장이 9.88에서 9.81로 거의 효과가 없었다.

State 용량만 늘리는 것보다 그 공간에 필요한 정보를 넣고 읽는 능력이 더 중요하다.

## 11. 한계와 주의점

### 긴 Context를 손실 없이 저장하는 것은 아니다

Mamba의 state 크기는 sequence length에 따라 늘지 않는다. 효율 면에서는 장점이지만 과거의 모든 세부 정보를 그대로 보존한다는 뜻은 아니다.

긴 context를 더 잘 활용했다는 실험 결과는 **학습된 압축이 task에 필요한 정보를 잘 남겼다**는 의미로 봐야 한다.

### Linear Time과 빠른 Wall-clock Time은 다르다

$O(L)$ complexity는 sequence가 길어질 때의 증가율을 설명한다. 실제 속도는 kernel 품질, batch size, state dimension, GPU memory hierarchy에 달려 있다. 논문에서도 selective scan을 직접 최적화했기 때문에 좋은 wall-clock result를 얻었다. Naive PyTorch recurrence만 구현하면 이론적 복잡도가 좋아도 느릴 수 있다.

### 모든 Data에서 Selection이 무조건 유리하지는 않다

논문은 이를 continuous-discrete spectrum의 **no free lunch**라고 설명한다. 고정된 dynamics를 가진 LTI SSM은 audio 같은 연속 신호에 잘 맞는 inductive bias가 있다. Input-dependent selection은 text와 DNA 같은 discrete data의 약점을 해결하지만, LTI bias가 유리한 문제에서는 오히려 방해가 될 수 있다.

### 논문의 최대 Language Model은 오늘날 기준으로 작다

핵심 language scaling experiment는 약 1.3B까지이고, 가장 큰 평가 모델은 약 2.8B 규모다. 논문도 더 큰 scale, instruction tuning, RLHF, quantization, in-context learning 같은 특성이 그대로 유지되는지를 향후 과제로 남긴다.

따라서 이 논문의 결과를 모든 대형 Transformer를 바로 대체할 수 있다는 의미로 읽으면 안 된다. **Linear-time recurrent model도 language에서 Transformer 수준으로 scaling할 가능성을 보였다**는 것이 더 정확하다.

## 12. 내가 이해한 핵심

Mamba 1을 처음 보면 attention을 없애고 SSM으로 바꾼 모델처럼 보인다. 하지만 논문을 읽고 나면 핵심은 단순히 attention을 제거한 것이 아니라는 점이 보인다.

전체 흐름을 다시 연결하면 다음과 같다.

```text
기존 SSM
고정된 dynamics 덕분에 convolution 가능
-> 빠르지만 입력 내용에 따라 기억할 정보를 고르기 어려움

Selective SSM
Δ, B, C를 입력의 함수로 만듦
-> token마다 기억, 삭제, 읽기를 선택
-> 대신 고정 convolution을 사용할 수 없음

Hardware-aware selective scan
parallel scan + kernel fusion + SRAM 계산 + recomputation
-> recurrence를 GPU에서 효율적으로 실행

Mamba block
selective SSM + local Conv + gate를 하나로 결합
-> Attention과 별도 MLP block이 없는 homogeneous architecture
```

기존 SSM은 convolution으로 빠르게 계산하기 위해 input-dependent dynamics를 포기했다. Mamba는 $\Delta$, $B$, $C$를 입력에 따라 바꾸면서 이 능력을 다시 가져왔다. 그리고 그 때문에 convolution을 사용할 수 없게 된 문제를 selective scan으로 해결했다.

내가 이해한 가장 큰 차이는 정보를 선택하는 시점이다.

Transformer는 과거를 KV cache에 보존한 뒤 query 시점에 필요한 위치를 찾는다. Mamba는 입력이 들어오는 시점에 무엇을 남길지 결정해 고정 크기의 state로 압축한다.

> Attention의 질문이 “과거 어디를 볼 것인가?”라면, Mamba의 질문은 “지금 들어온 정보 중 무엇을 미래에 남길 것인가?”다.

결국 Mamba 1은 **좋은 sequence model은 모든 정보를 저장하는 모델이 아니라, 필요한 정보를 잘 남기는 모델일 수 있다**는 것을 보여준다.

## 참고 자료

- Albert Gu, Tri Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), 2023/2024.
- [Official Mamba implementation](https://github.com/state-spaces/mamba)
