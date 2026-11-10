---
title: "FlashAttention-1 논문 정리: IO-Aware Exact Attention"
date: 2026-06-16 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - LLM
  - Transformer
  - Attention
  - GPU
source: "arXiv:2205.14135 - FlashAttention"
---

Transformer에서 attention은 핵심 연산이지만, sequence length가 길어질수록 비용이 빠르게 커진다.

기본 scaled dot-product attention은 다음과 같다.

$$
S = QK^T,\quad P = softmax(S),\quad O = PV
$$

여기서 `Q`, `K`, `V`는 각각 query, key, value이고, `S`와 `P`는 `N x N` 크기의 attention matrix다. 문제는 이 `N x N` 중간 matrix를 GPU memory에 실제로 저장하고 다시 읽는 과정이다.

FlashAttention 논문의 핵심은 간단하다.

> Attention을 근사하지 않고 정확하게 계산하되, GPU HBM read/write를 줄이도록 알고리즘을 다시 짜자.

즉 FlashAttention-1은 sparse attention이나 linear attention처럼 attention 자체를 근사하는 방법이 아니다. 결과는 standard attention과 같고, 계산 순서와 memory access 방식을 바꾼다.

![FlashAttention 원문 Figure: runtime과 sparsity speedup]({{ '/assets/images/blog/flashattention-paper-figure.png' | relative_url }})

*원 논문 Figure 2는 standard attention과 FlashAttention의 HBM access, runtime 차이를 보여준다.*

## 왜 FLOPs만 보면 부족한가

딥러닝 연산 성능을 볼 때 흔히 FLOPs를 먼저 본다. 하지만 GPU에서는 FLOPs만으로 실제 속도가 결정되지 않는다.

GPU memory hierarchy는 대략 다음처럼 볼 수 있다.

- HBM: GPU global memory. 용량은 크지만 상대적으로 느리다.
- SRAM/shared memory: GPU chip 안의 작은 memory. 용량은 작지만 훨씬 빠르다.
- Register: thread가 직접 쓰는 가장 빠른 작은 저장 공간.

논문은 A100 기준으로 HBM bandwidth는 약 `1.5-2.0TB/s`, on-chip SRAM bandwidth는 약 `19TB/s`라고 설명한다. SRAM은 HBM보다 훨씬 빠르지만, 크기가 매우 작다.

그래서 attention을 빠르게 만들려면 단순히 곱셈 수만 줄이는 것이 아니라, HBM에 무엇을 쓰고 다시 읽는지도 봐야 한다. 논문은 이 관점을 **IO-aware**라고 부른다.

## Standard Attention의 병목

일반적인 attention 구현은 다음 순서로 동작한다.

```text
Q, K를 읽어 S = QK^T 계산
S를 HBM에 저장
S를 다시 읽어 P = softmax(S) 계산
P를 HBM에 저장
P와 V를 읽어 O = PV 계산
O를 HBM에 저장
```

여기서 `S`와 `P`가 모두 `N x N` matrix다.

sequence length가 `N = 4096`이면 `S` 하나만 해도 약 1,677만 개 원소를 가진다. batch와 head까지 붙으면 훨씬 커진다.

더 나쁜 점은 `S`와 `P`를 한 번 만드는 데서 끝나지 않는다는 것이다. masking, softmax, dropout, backward pass를 위해 중간 값을 HBM에 저장하고 다시 읽는다. Attention의 이론적 계산량도 크지만, 실제 runtime에서는 이 HBM traffic이 큰 병목이 된다.

논문에서 비교한 GPT-2 medium 설정에서는 standard attention과 FlashAttention이 다음처럼 달랐다.

| 항목 | Standard Attention | FlashAttention |
| --- | ---: | ---: |
| GFLOPs | 66.6 | 75.2 |
| HBM read/write | 40.3GB | 4.4GB |
| Runtime | 41.7ms | 7.3ms |

흥미로운 점은 FlashAttention의 FLOPs가 오히려 더 많다는 것이다. 그런데 HBM read/write가 훨씬 줄어서 실제 runtime은 더 짧다.

이게 논문의 중요한 메시지다.

> GPU에서는 계산을 조금 더 하더라도 memory traffic을 크게 줄이면 더 빨라질 수 있다.

## FlashAttention의 핵심 아이디어

FlashAttention은 `N x N` attention matrix 전체를 HBM에 만들지 않는다.

대신 `Q`, `K`, `V`를 block으로 나누고, 작은 block을 SRAM에 올려 계산한다. 각 block 계산이 끝나면 필요한 output과 softmax 통계만 유지하고, 거대한 attention matrix는 저장하지 않는다.

논문 Algorithm 1의 흐름을 단순화하면 다음과 같다.

```text
K, V를 block으로 나눈다.
Q를 block으로 나눈다.

for each K_j, V_j block:
  K_j, V_j를 HBM에서 SRAM으로 load

  for each Q_i block:
    Q_i와 현재까지의 O_i, m_i, l_i를 SRAM으로 load
    S_ij = Q_i K_j^T 를 SRAM 안에서 계산
    block 단위 row max와 exp sum을 계산
    기존 O_i를 새 softmax normalization 기준으로 보정
    새 block의 P_ij V_j 결과를 O_i에 누적
    O_i, m_i, l_i를 HBM에 write
```

여기서 `m_i`와 `l_i`가 중요하다.

- `m_i`: 지금까지 본 score의 row-wise maximum
- `l_i`: softmax denominator의 누적값
- `O_i`: 지금까지 누적된 output block

즉 FlashAttention은 attention score 전체를 저장하는 대신, softmax를 이어서 계산하는 데 필요한 작은 통계만 저장한다.

![FlashAttention의 memory hierarchy와 IO-aware attention 구조]({{ '/assets/images/blog/flashattn.png' | relative_url }})

*FlashAttention은 Q, K, V block을 SRAM에 올려 계산하고, `N x N` attention matrix를 HBM에 materialize하지 않는다.*

## Online Softmax

FlashAttention에서 가장 까다로운 부분은 softmax다.

Softmax는 한 row의 모든 score를 보고 normalization해야 한다.

$$
softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

그런데 FlashAttention은 score row 전체를 한 번에 들고 있지 않는다. block 단위로 `S_ij`만 본다. 단순히 block마다 따로 softmax를 하면 전체 softmax와 다른 결과가 나온다.

이를 해결하기 위해 논문은 online softmax를 사용한다. 새 block이 들어올 때마다 row maximum과 denominator를 갱신한다.

기존까지의 통계를 `m`, `l`이라고 하고, 새 block의 통계를 `m_tilde`, `l_tilde`라고 하면 새 maximum은 다음과 같다.

$$
m_{new} = max(m, \tilde{m})
$$

denominator는 maximum이 바뀐 만큼 기존 값을 보정해서 더한다.

$$
l_{new}
= e^{m - m_{new}}l
+ e^{\tilde{m} - m_{new}}\tilde{l}
$$

output도 같은 방식으로 보정한다.

$$
O_{new}
= \frac{
e^{m - m_{new}}lO
+ e^{\tilde{m} - m_{new}}\tilde{P}V
}{
l_{new}
}
$$

이렇게 하면 block 단위로 계산해도 전체 row를 한 번에 softmax한 것과 같은 결과를 얻는다.

## Backward Pass와 Recomputation

학습에서는 forward만 빠르면 부족하다. Backward pass도 memory를 많이 쓴다.

일반 attention은 backward를 위해 `S` 또는 `P` 같은 `N x N` 중간 matrix를 저장한다. 이 값들이 없으면 gradient 계산에 필요하기 때문이다.

FlashAttention은 이 중간 matrix를 저장하지 않는다. 대신 forward에서 다음 값만 저장한다.

- output `O`
- row-wise maximum `m`
- softmax denominator `l`

Backward 때는 `Q`, `K`, `V` block과 `m`, `l`을 사용해 필요한 attention block을 SRAM 안에서 다시 계산한다.

이 방식은 recomputation이다. 계산량은 늘어난다. 하지만 `N x N` matrix를 HBM에서 읽고 쓰는 비용이 사라지기 때문에, 실제로는 backward도 더 빨라질 수 있다.

논문은 이를 selective gradient checkpointing처럼 볼 수 있다고 설명한다. 일반 gradient checkpointing은 memory를 줄이는 대신 느려지는 경우가 많지만, FlashAttention에서는 HBM access 감소가 커서 recomputation이 오히려 유리해진다.

## IO Complexity

FlashAttention은 dense exact attention이므로 FLOPs 자체는 여전히 다음과 같다.

$$
O(N^2d)
$$

즉 query-key 조합 수가 사라지는 것은 아니다.

하지만 HBM access는 줄어든다. 논문은 SRAM 크기를 `M`, head dimension을 `d`, sequence length를 `N`이라고 할 때 다음처럼 분석한다.

| 항목 | HBM access |
| --- | --- |
| Standard Attention | $\Theta(Nd + N^2)$ |
| FlashAttention | $\Theta(N^2 d^2 / M)$ |

일반적으로 `d`는 64 또는 128 정도이고, `M`은 SRAM 크기다. `d^2`보다 `M`이 충분히 크면 FlashAttention은 standard attention보다 훨씬 적은 HBM access를 사용한다.

또한 FlashAttention은 input과 output 외 추가 memory가 `O(N)`이다. 반면 standard attention은 `S`, `P` 때문에 `O(N^2)` memory를 사용한다.

정리하면 다음과 같다.

| 항목 | Standard Attention | FlashAttention |
| --- | --- | --- |
| 결과 | Exact attention | Exact attention |
| FLOPs | $O(N^2d)$ | $O(N^2d)$ |
| 추가 memory | $O(N^2)$ | $O(N)$ |
| 병목 | `N x N` matrix HBM read/write | block 계산과 누적 |
| 핵심 기법 | 중간 matrix materialization | tiling, online softmax, recomputation |

## Kernel Fusion

FlashAttention은 단순한 수식 변형만이 아니다. 구현 관점에서는 CUDA kernel fusion이 중요하다.

일반 구현에서는 다음 연산들이 여러 kernel로 나뉘기 쉽다.

- matrix multiply
- masking
- softmax
- dropout
- matrix multiply

각 kernel 사이에서 중간 결과가 HBM에 쓰이고 다시 읽힌다.

FlashAttention은 tiling 구조 덕분에 이 과정을 하나의 CUDA kernel 안에서 처리할 수 있다. `Q`, `K`, `V` block을 SRAM에 올리고, score 계산, masking, softmax, dropout, value 곱을 이어서 수행한 뒤 최종 output만 HBM에 쓴다.

그래서 FlashAttention은 "attention 공식을 바꾼 방법"이라기보다 "attention을 GPU memory hierarchy에 맞게 다시 구현한 방법"에 가깝다.

## 실험 결과

논문은 FlashAttention이 단순 microbenchmark뿐 아니라 실제 model training에서도 효과가 있다고 보인다.

대표 결과는 다음과 같다.

- BERT-large training에서 MLPerf 1.1 기록 대비 약 15% 빠름
- GPT-2 training에서 Hugging Face 구현 대비 최대 3x speedup
- Long Range Arena에서 standard attention 대비 약 2.4x speedup
- common sequence length에서 standard attention 대비 최대 3x 빠름
- attention memory footprint는 sequence length에 선형으로 증가
- exact attention baseline 대비 최대 20x memory efficient

또한 FlashAttention 덕분에 더 긴 context로 학습할 수 있었다.

GPT-2 small 실험에서는 context length를 1K에서 4K로 늘려도 Megatron-LM의 1K context baseline보다 빠르게 학습했고, perplexity는 0.7 좋아졌다. Long document classification에서도 sequence length를 늘리면서 성능이 좋아졌다.

논문에서 인상적인 결과 중 하나는 Path-X다. 기존 Transformer 계열은 memory 부족이나 random 수준 성능 때문에 어려웠는데, FlashAttention으로 sequence length 16K를 처리해 61.4% accuracy를 보고했다.

## Block-Sparse FlashAttention

논문은 FlashAttention을 block-sparse attention으로도 확장한다.

Dense FlashAttention은 exact attention이다. 모든 query-key 조합을 본다. 반면 block-sparse FlashAttention은 attention matrix의 일부 block만 계산한다.

이 경우 attention 자체는 sparse mask에 의해 근사 또는 제한된 attention이 된다. 하지만 FlashAttention의 IO-aware tiling 구조를 그대로 활용할 수 있다.

논문에 따르면 nonzero block 비율을 `s`라고 하면 block-sparse FlashAttention의 HBM access는 대략 다음처럼 줄어든다.

$$
\Theta(Nd + N^2 d^2 M^{-1}s)
$$

즉 sparsity가 높을수록 HBM access와 runtime이 줄어든다. 다만 이 글에서 말하는 FlashAttention-1의 핵심은 dense exact attention이고, block-sparse 버전은 그 확장으로 보는 것이 좋다.

## 한계

FlashAttention은 매우 강력하지만, 모든 문제를 해결하지는 않는다.

첫째, dense attention의 계산 복잡도 자체를 없애지는 않는다. `N x N` score를 저장하지 않을 뿐, dense exact attention이라면 query-key 조합은 여전히 계산한다.

둘째, CUDA kernel 수준의 구현이 필요하다. 논문도 새로운 attention 변형마다 CUDA kernel을 직접 작성해야 하는 점을 한계로 언급한다. 또한 GPU architecture가 바뀌면 구현 최적화가 그대로 이전되지 않을 수 있다.

셋째, LLM serving의 decode 단계에서는 병목이 다를 수 있다. FlashAttention은 여러 token을 한 번에 처리하는 training이나 prompt prefill에서 특히 중요하다. 반면 decode는 token을 하나씩 생성하는 경우가 많고, 이때는 KV cache memory, scheduler, batching이 더 큰 병목이 될 수 있다.

## FlashAttention과 PagedAttention 차이

FlashAttention과 PagedAttention은 이름이 비슷해서 헷갈리지만, 해결하는 문제가 다르다.

| 항목 | FlashAttention | PagedAttention |
| --- | --- | --- |
| 대상 | attention 계산 kernel | KV cache memory 관리 |
| 핵심 문제 | HBM read/write 비용 | serving 중 cache fragmentation과 낭비 |
| 주요 상황 | training, prefill, 긴 sequence attention | decode, multi-request serving |
| 대표 시스템 | FlashAttention CUDA kernel | vLLM |

FlashAttention은 `QK^T -> softmax -> PV` 계산을 더 memory-efficient하게 만든다. PagedAttention은 생성 중 쌓이는 KV cache를 block 단위로 관리한다.

둘은 경쟁 기술이 아니라 서로 다른 병목을 줄이는 기술이다.

## 내가 이해한 핵심

FlashAttention-1의 핵심은 attention을 **IO-aware algorithm**으로 다시 보는 것이다.

기존 attention은 `N x N` attention matrix를 HBM에 materialize한다. 이 때문에 sequence length가 길어질수록 memory footprint와 HBM traffic이 커진다.

FlashAttention은 tiling으로 `Q`, `K`, `V`를 block 단위로 SRAM에 올리고, online softmax로 정확한 softmax 결과를 유지하며, backward에서는 필요한 attention block을 recompute한다. 그 결과 dense exact attention을 유지하면서 추가 memory를 `O(N^2)`에서 `O(N)`으로 줄이고, HBM access를 크게 낮춘다.

가장 중요한 문장은 이것이다.

> FlashAttention은 attention을 덜 계산하는 방법이 아니라, attention을 덜 옮기는 방법이다.

그래서 FLOPs만 보면 FlashAttention의 장점이 잘 보이지 않는다. 하지만 GPU에서는 memory movement가 runtime을 결정하는 경우가 많고, FlashAttention은 바로 그 지점을 찌른다.

## 참고 자료

- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention PDF](https://arxiv.org/pdf/2205.14135)
