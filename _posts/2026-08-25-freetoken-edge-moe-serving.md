---
title: "FreeToken 논문 정리: GPU·CPU·PCIe를 함께 쓰는 엣지 MoE 서빙"
date: 2026-08-25 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
summary: "FreeToken이 거대한 MoE 모델을 개인용 PC에서 실행하기 위해 prefill과 decode를 다르게 최적화하고, GPU cache miss를 PCIe 전송과 CPU 계산으로 나누는 원리를 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - MoE
  - Inference
  - Serving
  - Edge AI
  - CPU GPU Offloading
  - FreeToken
  - Paper
source: "arXiv:2608.16157"
---

# FreeToken: 개인용 PC를 하나의 MoE 추론 시스템으로 만들기

이 글은 논문 [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)을 바탕으로 정리한 글이다.

> Shuo Yang, Xiaoze Fan, Melissa Pan, Haocheng Xi, Zhe Wang, Shanlin Sun, Kurt Keutzer, Song Han, Matei Zaharia, Chenfeng Xu, Ion Stoica  
> [[Paper](https://arxiv.org/pdf/2608.16157)] [[Code](https://github.com/FlashML-org/FreeToken)]

핵심 관심사는 하나다.

> Open-weight 모델을 내려받을 수 있다는 것과 개인이 그 모델을 실제로 돌릴 수 있다는 것은 다르다.

거대한 Mixture-of-Experts(MoE) 모델은 token마다 일부 expert만 활성화하므로 계산량이 전체 모델 크기에 비해 작다. 하지만 **전체 expert weight는 여전히 어딘가에 저장되어 있어야 한다.** 수백 GB에 이르는 expert pool이 VRAM에 들어가지 않으면 CPU DRAM에 두고 필요할 때 가져와야 한다. 이때 병목은 단순 GPU 연산이 아니라 PCIe 전송, DRAM bandwidth, CPU 계산, VRAM cache 사이의 조율로 옮겨간다.

FreeToken은 개인용 PC를 작은 GPU 한 장으로 보지 않는다. GPU, CPU, DRAM, PCIe를 묶은 **하나의 이기종 추론 플랫폼**으로 보고, 실행 단계와 현재 자원 상태에 따라 일을 다시 배분한다.

한 줄로 요약하면 이렇다.

> FreeToken은 prefill에서는 expert 전송과 GPU 계산을 겹치고, decode에서는 GPU cache miss를 PCIe 전송과 CPU 계산으로 나눠 처리하며, agent가 문맥을 수정해도 가능한 prefix state를 재사용하는 엣지 MoE 서빙 시스템이다.

## 1. 왜 MoE가 로컬 추론에 유리한가

Dense 모델은 매 token마다 거의 모든 parameter를 사용한다. 반면 MoE layer에는 여러 expert가 있지만 router가 token마다 일부 expert만 선택한다.

예를 들어 논문이 다루는 DeepSeek-V4-Flash는 284B parameter 모델이지만 한 token에서 활성화되는 parameter는 약 13B다. 각 MoE layer에서 256개 routed expert 가운데 6개만 선택되기 때문이다.

```text
Dense model
token -> 전체 weight 사용

MoE model
token -> router -> 일부 expert만 사용
```

이 sparse activation 덕분에 한 token을 계산하는 데 필요한 FLOPs와 활성 parameter footprint는 전체 모델 크기보다 훨씬 작다. 284B 모델이라도 활성 경로만 보면 RTX 5090의 32 GB VRAM으로 계산할 가능성이 생긴다.

그러나 **활성 parameter가 작다는 것이 전체 model weight가 작다는 뜻은 아니다.** 다음 token이 어떤 expert를 고를지 미리 확정할 수 없으므로 전체 expert pool은 CPU DRAM이나 storage에 남아 있어야 한다.

```text
GPU VRAM
├─ non-expert weight
├─ KV cache
└─ 자주 쓰는 expert cache

CPU DRAM
└─ 전체 expert pool (source of truth)
```

결국 로컬 MoE 서빙의 질문은 “모델이 GPU에 전부 들어가는가?”에서 다음 질문으로 바뀐다.

```text
지금 필요한 expert가 GPU에 있는가?
없다면 PCIe로 가져올 것인가, CPU에서 바로 계산할 것인가?
제한된 VRAM을 KV cache와 expert cache에 어떻게 나눌 것인가?
```

FreeToken은 이 세 질문에 runtime 수준의 답을 제시한다.

## 2. Prefill과 Decode는 서로 다른 문제다

LLM inference는 크게 prefill과 decode로 나뉜다.

| 단계 | 처리 방식 | MoE에서 나타나는 병목 |
| --- | --- | --- |
| Prefill | 긴 prompt token을 병렬 처리 | 여러 token의 route 합집합이 거의 모든 expert를 활성화 |
| Decode | 새 token을 한 step씩 생성 | 매 step의 expert cache miss와 memory bandwidth |

Decode에서는 한 token이 몇 개 expert만 선택하므로 MoE sparsity가 잘 유지된다. 하지만 수천 token을 한 번에 처리하는 prefill에서는 각 token이 서로 다른 expert를 고른다. 모든 route의 합집합을 보면 layer의 expert 대부분에 접근하게 된다.

즉 같은 모델도 실행 단계에 따라 성격이 반대로 바뀐다.

```text
Prefill: token은 많고, expert working set은 거의 dense
Decode : token은 적고, expert working set은 sparse
```

이 차이를 무시하고 하나의 offloading 정책만 사용하면 어느 한쪽에서 손해가 커진다. FreeToken은 prefill과 decode에 서로 다른 전략을 사용한다.

## 3. Prefill: 전체 layer 전송을 GPU 계산과 겹치기

### On-demand loading이 느린 이유

Prefill은 결국 layer의 거의 모든 expert를 사용한다. 이 상황에서 router 결과가 나온 뒤 필요한 expert를 하나씩 요청하면 PCIe 전송이 잘게 쪼개지고, GPU는 weight가 도착할 때까지 기다리게 된다.

DeepSeek-V4-Flash의 FP4 expert pool은 약 140 GB다. 논문에 따르면 이를 한 차례 전송하는 데 RTX 5090의 PCIe 5.0 ×16에서는 약 2초, RTX 4090/3090의 PCIe 4.0 ×16에서는 약 5초, laptop의 ×8 link에서는 10초 이상 걸릴 수 있다.

### Full-layer double buffering

FreeToken은 prefill에서 어차피 대부분의 expert가 필요하다는 점을 받아들인다. 그리고 GPU memory에 두 layer 분량의 buffer를 만든다.

```text
시간 t
GPU compute : layer l
PCIe load   : layer l+1

시간 t+1
GPU compute : layer l+1
PCIe load   : layer l+2
```

GPU가 현재 layer를 계산하는 동안 별도의 transfer stream이 다음 layer의 expert 전체를 미리 읽는다. 다음 layer의 routing 결과를 기다릴 필요가 없으므로 전송을 일찍 시작할 수 있다. 계산이 끝나면 두 buffer가 역할을 바꾼다.

![FreeToken의 full-layer double buffering과 semantic state checkpoint]({{ '/assets/images/blog/freetoken-prefill-pipeline-state-checkpoint.png' | relative_url }})

*Prefill에서는 layer 단위 double buffering으로 PCIe 전송과 GPU 계산을 겹친다. Agent가 tool output이나 reasoning block을 제거하면 semantic anchor의 checkpoint에서 재개해 바뀐 suffix만 다시 prefill한다. 출처: FreeToken Figure 2 일부.*

이 방식의 목적은 expert 전송을 없애는 것이 아니다. **전송 시간 뒤에 GPU 계산 시간을 숨기는 것**이다. Buffer 두 개를 확보할 VRAM이 없으면 FreeToken은 memory를 과도하게 할당하지 않고 on-demand loading으로 fallback한다.

또한 prefill과 decode가 별도의 expert cache를 사용하지 않는다. 같은 slot pool을 공유하므로 prefill 뒤에 남은 expert가 decode의 초기 cache를 채우는 효과도 생긴다.

## 4. Agent 문맥 편집 뒤에도 prefix state를 재사용하기

Agent workload는 일반적인 단일 대화보다 prefill을 훨씬 자주 일으킨다.

Coding agent와 tool-using agent는 매 turn마다 다음과 같은 작업을 한다.

- 오래된 tool output을 placeholder로 교체한다.
- 이전 reasoning 또는 thinking block을 제거한다.
- 최근 몇 개 observation만 남기고 앞부분을 줄인다.
- system prompt와 tool definition은 유지한 채 새 결과를 붙인다.

Standard full attention만 있다면 공통 prefix의 KV cache를 radix tree로 찾아 재사용할 수 있다. 하지만 최근 모델은 sliding-window attention이나 recurrent layer를 함께 사용하는 hybrid architecture가 많다.

Recurrent layer는 과거 전체를 하나의 state로 압축한다.

$$
s_t = f(s_{t-1}, x_t)
$$

이 state는 KV cache처럼 token별 조각을 부분적으로 잘라 쓰기 어렵다. 문맥 중간이 바뀌면 변경 지점 뒤의 state는 모두 무효가 된다. Checkpoint를 촘촘히 저장하면 좋지만, recurrent state 하나가 수백 token의 KV cache만큼 클 수 있어 많이 보관하기도 어렵다.

FreeToken은 checkpoint 위치를 균일한 token 간격으로 정하지 않고 **semantic anchor**에 둔다.

```text
[system]
   ▲ checkpoint
[reasoning] </think>
   ▲ checkpoint
[tool call] </tool_call>
   ▲ checkpoint
[tool output] </tool_output>
   ▲ checkpoint
[answer]
```

`</think>`, `</tool_call>`, `</tool_output>` 같은 special token 경계는 agent framework가 block을 제거하거나 교체하는 위치와 일치한다. 문맥이 수정되어도 편집 지점 직전의 semantic boundary까지는 보존될 가능성이 높다.

새 요청이 들어오면 FreeToken은 살아남은 가장 깊은 checkpoint를 복원한다. Full-attention layer는 같은 지점까지 KV cache를 재사용하고, recurrent layer는 저장된 state에서 다시 시작한다. 결과적으로 전체 context가 아니라 새 suffix만 re-prefill한다.

이 아이디어의 핵심은 cache 용량뿐 아니라 **어디에 checkpoint를 놓는가**도 중요하다는 것이다. Agent의 문맥 구조를 모르는 임의의 checkpoint보다 tool과 turn의 경계를 아는 checkpoint가 살아남을 확률이 높다.

## 5. Decode: Expert cache는 현재 routing을 따라가야 한다

Decode에서 한 token은 일부 expert만 선택한다. 이때 가장 빠른 경우는 선택된 expert가 이미 VRAM에 있는 cache hit다.

기존의 static placement는 model load 또는 prefill 시점에 “자주 쓸 것 같은 expert”를 GPU에 고정한다. 하지만 routing은 token과 workload에 따라 계속 달라진다. Prefill에서 많이 쓰인 expert가 이후 decode에서도 계속 hot하다는 보장은 없다.

FreeToken은 모든 MoE layer가 공유하는 LRU expert cache를 사용한다.

```text
cache hit  -> recency 갱신
cache fill -> 새 expert를 slot에 저장
cache full -> 가장 오래 사용하지 않은 expert 축출
```

연속 token은 같은 layer에서 겹치는 expert를 선택하는 경향이 있다. 논문은 이를 **temporal expert locality**로 설명한다. Cache가 특정 workload에 맞춰 고정되는 대신, 실제 router가 최근 선택한 expert를 따라 움직이게 하는 것이다.

하지만 cache가 아무리 좋아도 cold start, route 변화, 작은 VRAM 때문에 miss는 남는다. FreeToken의 가장 중요한 설계는 이 남은 miss를 처리하는 방식이다.

## 6. Cache miss를 PCIe와 CPU로 나누는 `q*` 정책

GPU에 없는 expert를 처리하는 방법은 두 가지다.

1. CPU DRAM의 expert를 PCIe로 GPU에 옮긴 뒤 계산한다.
2. Expert가 있는 CPU에서 바로 계산하고 작은 activation/output만 주고받는다.

첫 번째 방법은 GPU가 빠르고 expert가 cache에 남아 미래 hit도 만들지만 PCIe bandwidth가 필요하다. 두 번째 방법은 큰 expert weight를 옮기지 않아도 되지만 CPU의 DRAM bandwidth와 계산 속도에 제한된다.

FreeToken은 한쪽만 선택하지 않는다. 한 step에서 발생한 `m`개의 unique miss를 두 집합으로 나눈다.

$$
M = F \dot{\cup} C, \qquad q = |F|
$$

- $F$: PCIe로 GPU cache에 채운 뒤 GPU에서 계산할 expert
- $C$: CPU에 둔 채 바로 계산할 expert
- $q$: GPU로 옮길 expert 수

![FreeToken의 decode-time bandwidth-adaptive execution]({{ '/assets/images/blog/freetoken-decode-bandwidth-adaptive-execution.png' | relative_url }})

*예시에서는 12개 routed expert 중 8개가 LRU cache hit다. 남은 4개 가운데 1개는 GPU로 옮기고 3개는 CPU에서 계산한다. 두 partial output은 마지막에 정확히 합쳐진다. 출처: FreeToken Figure 2 일부.*

### 왜 `q* = m B_P / B_H`인가

기호를 다음처럼 두자.

- $S$: expert 하나의 byte 크기
- $B_P$: 측정된 PCIe expert transfer bandwidth
- $B_H$: 측정된 CPU-side expert processing bandwidth

PCIe DMA도 CPU 계산도 같은 host memory에서 expert weight를 읽는다. PCIe가 $B_P$만큼 사용 중이면 CPU 계산에 남는 bandwidth는 다음과 같다.

$$
B_R = \max(B_H - B_P, 0)
$$

GPU로 보낼 $q$개 expert와 CPU에서 처리할 $m-q$개 expert의 시간은 대략 다음과 같다.

$$
T_{fill}(q) \approx \frac{qS}{B_P}
$$

$$
T_{cpu}(m-q) \approx \frac{(m-q)S}{B_H-B_P}
$$

두 branch는 동시에 실행되므로 전체 layer latency는 둘 중 느린 쪽에 의해 결정된다. 두 시간이 비슷하도록 맞추면 다음 식을 얻는다.

$$
q^* \approx m\frac{B_P}{B_H}
$$

예를 들어 miss가 4개이고 측정 결과가 $B_P:B_H = 1:4$라면 다음과 같다.

$$
q^* = 4 \times \frac{1}{4} = 1
$$

따라서 1개는 PCIe로 옮기고 3개는 CPU에서 처리한다. PCIe와 CPU가 각자 맡은 일을 비슷한 시점에 끝내도록 하는 배분이다.

반대로 $B_H$가 $B_P$에 가까운 시스템은 PCIe 전송 뒤 CPU에 남는 bandwidth가 거의 없다. 이때 $q^*$는 $m$에 가까워지고 사실상 모든 miss를 GPU cache fill로 처리한다.

중요한 점은 이 비율을 제품 사양표에서 가져오지 않는다는 것이다. FreeToken은 실제 배포할 tensor shape와 hardware에서 $B_P$, $B_H$를 측정한다. 같은 RTX 5090이라도 host DRAM과 CPU 구성이 다르면 정책이 달라질 수 있다.

### 근사 계산이 아니라 정확한 실행

CPU와 GPU는 서로 다른 expert의 partial sum을 계산한다. 이후 gate weight가 반영된 결과를 합쳐 원래 MoE layer output을 만든다.

```text
y = y_GPU + y_CPU
```

Expert를 생략하거나 값싼 대체 expert로 바꾸는 것이 아니다. 논문이 말하는 exact merge는 같은 routed expert와 같은 weight를 사용하되 실행 위치만 나눈다는 뜻이다.

또한 FreeToken은 cache가 계속 warm-up되도록 최소 한 개의 miss는 GPU cache에 채운다. 당장 CPU 실행이 유리해 보여도 모든 miss를 CPU로 보내면 미래 cache hit를 만들 수 없기 때문이다.

## 7. Dynamic한 LRU를 CUDA Graph 안에 넣기

Expert cache는 매 token, 매 layer마다 miss 수와 eviction 대상이 달라진다. 이를 Python이나 CPU scheduler가 매번 결정하면 GPU와 host 사이의 synchronization이 다시 병목이 된다.

FreeToken은 routing에 의존하는 control을 GPU에 둔다.

- routed expert를 deduplicate한다.
- residency table에서 hit와 miss를 분류한다.
- bandwidth 기반으로 $q$를 계산한다.
- LRU victim을 고른다.
- logical expert ID를 physical slot ID 또는 CPU assignment flag로 바꾼다.

Dynamic한 결정을 graph 구조 변경이 아니라 fixed-shape buffer 안의 **data와 valid count**로 표현한다. 그래서 정적으로 capture한 CUDA Graph를 유지할 수 있다.

CPU branch도 같은 graph에 포함된다. Stable pinned buffer, persistent task descriptor, host-function submit node, GPU branch, synchronization, result copy를 함께 capture한다. 매 token마다 Python이 CPU worker에게 새 작업을 조립해 보내지 않아도 된다.

LRU victim을 고를 때도 miss 하나마다 전체 cache를 다시 scan하지 않는다. 한 번의 pass로 최대 `K`개의 least-recently-used candidate를 찾고, 실제 miss 수에 맞춰 앞의 `q`개 slot만 사용한다.

이 구현은 논문의 간단한 수식이 실제 serving 성능으로 이어지기 위해 필요한 부분이다. 정책 계산이 싸더라도 매 layer마다 host synchronization을 만들면 그 이득을 쉽게 잃을 수 있다.

## 8. VRAM은 실행 중에도 다시 나눠야 한다

Datacenter GPU는 보통 serving process가 전용으로 사용하지만 개인용 GPU는 browser, desktop compositor, game과 VRAM을 공유한다. 사용 가능한 memory가 실행 중에도 바뀔 수 있다.

Agent session이 길어지면 필요한 KV cache는 커지는 반면 expert working set은 비슷하게 유지된다. 첫 turn에 정한 KV cache와 expert cache의 비율이 20번째 turn에도 최적이라는 보장이 없다.

FreeToken에서 CPU DRAM의 전체 expert pool은 source of truth다. GPU expert cache는 없어져도 정확도가 바뀌지 않고 성능만 달라진다. 이 특성을 이용해 scheduler safe point에서 GPU expert cache를 새 VRAM budget에 맞게 다시 만들 수 있다.

```text
초기 turn
[expert cache 크게][KV cache 작게]

긴 context가 쌓인 뒤
[expert cache 작게][KV cache 크게]
```

Engine을 재시작하거나 host의 전체 expert pool을 다시 읽지 않고도 memory split을 조정한다.

Startup도 줄인다. FreeToken Weight(FTW) format은 model별 checkpoint를 runtime이 사용할 expert-bank layout으로 미리 정규화한다. Disk에서 최종 host layout으로 바로 읽어 buffer를 채운 뒤 memory를 pin한다. 이 순서로 빈 buffer를 먼저 pin할 때 생기는 page fault와 zero-fill 비용을 피한다. GPU cache는 별도 warm-up을 강제하지 않고 첫 요청의 일반적인 miss 처리 과정에서 자연스럽게 채운다.

## 9. 실험 환경

논문은 RTX 4060 laptop부터 RTX PRO 6000 workstation까지 6개 system을 사용한다. PCIe bandwidth $B_P$와 CPU-side bandwidth $B_H$는 spec sheet가 아니라 실제 tensor shape로 측정했다.

![FreeToken 실험에 사용한 hardware와 측정 bandwidth]({{ '/assets/images/blog/freetoken-test-systems.png' | relative_url }})

*같은 GPU라도 host 구성이 다르면 $B_P$와 $B_H$의 균형이 달라진다. 5090 server와 5090 desktop을 별도로 비교한 이유다. 출처: FreeToken Table 1.*

주요 model은 다음과 같다.

| 모델 | 전체 / 활성 parameter | Expert precision | 주요 hardware |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 35B / 약 3B | BF16, laptop은 NVFP4 | RTX 4060–5090 |
| DeepSeek-V4-Flash | 284B / 13B | MXFP4 | RTX 5090 |
| GLM-5.2 | 753B / 40B | NVFP4 | RTX PRO 6000 96 GB |

Workload는 단순한 synthetic token generation이 아니라 네 가지 agent scenario를 사용한다.

1. AIME math reasoning: single-turn, long chain-of-thought
2. OpenCode coding agent: SWE-bench issue와 실제 tool execution
3. Claude Code coding agent: subagent의 concurrent request와 56–65K token session
4. OpenClaw email/calendar agent: 약 24.5K token system context와 13개 turn

Baseline은 llama.cpp, Ollama, KTransformers, MoE-Infinity다. Qwen3.6은 모든 engine에서 BF16을 사용했고, DeepSeek-V4-Flash는 모든 engine이 native MXFP4 expert block을 bit-identical하게 사용했다.

## 10. End-to-End 결과

RTX 5090에서 Qwen3.6-35B-A3B와 DeepSeek-V4-Flash를 네 workload로 실행한 결과는 다음과 같다.

![FreeToken end-to-end serving 결과]({{ '/assets/images/blog/freetoken-end-to-end-results.png' | relative_url }})

*위쪽은 decode throughput, 아래쪽은 mean TTFT다. × 표시는 해당 engine이 그 configuration을 실행하지 못했다는 뜻이다. 출처: FreeToken Figure 3.*

### Decode throughput

FreeToken은 RTX 5090에서 다음 성능을 보고한다.

- Qwen3.6-35B-A3B: 77–83 tok/s
- DeepSeek-V4-Flash: 22–25 tok/s
- 각 workload의 가장 강한 baseline 대비 각각 1.8–2.3배, 1.5–1.9배

Agent workload가 복잡해져도 single-turn W1 대비 decode rate 변화가 12% 이내였다. 반면 일부 baseline은 context와 agent pattern에 따라 더 크게 느려졌다. 논문이 single-stream benchmark만으로 agent serving 성능을 판단하면 안 된다고 강조하는 이유다.

### TTFT와 tail latency

Mean TTFT에서는 FreeToken이 6개의 multi-turn model-workload 조합 중 5개에서 가장 낮았다. Qwen3.6의 W3에서는 KTransformers의 GPU prefill path가 더 빨랐고, 짧은 single-turn prompt인 W1에서는 llama.cpp가 유리했다.

더 중요한 결과는 worst-case TTFT다.

- FreeToken: 모든 cell에서 44초 미만
- llama.cpp: 최대 232초
- Ollama: 최대 179초
- KTransformers: 최대 946초

Agent client에는 idle watchdog 또는 request timeout이 있다. 따라서 tail TTFT는 단지 평균 latency보다 조금 나쁜 문제가 아니라, tool workflow 전체가 timeout으로 실패할 수 있는 availability 문제다.

## 11. 어떤 설계가 성능을 만들었는가

![FreeToken prefill pipeline과 expert cache miss 분석]({{ '/assets/images/blog/freetoken-prefill-cache-breakdown.png' | relative_url }})

*왼쪽은 prompt 길이에 따른 prefill throughput, 오른쪽은 expert cache 크기에 따른 decode miss rate다. 출처: FreeToken Figure 4.*

### Double buffering의 효과

Qwen3.6 BF16, RTX 5090에서 8,192-token prefill chunk는 1.19–1.22초가 걸렸다. 64.4 GB expert pool을 측정 PCIe bandwidth 52.7 GB/s로 한 번 보내는 시간에 가깝다. 즉 GPU expert 계산이 전송 뒤에 거의 숨겨지고 prefill이 transfer-bound 상태에 도달한 것이다.

두 번째 buffer를 제거해 전송과 계산을 직렬화하면 throughput은 prompt 4K에서 19%, 8K에서 25%, 16K에서 26% 감소했다. Prompt가 길수록 overlap으로 숨길 수 있는 계산 비중이 커진다.

### LRU expert cache의 효과

같은 routing trace와 같은 cache capacity에서 placement policy만 비교했다.

| 모델과 cache 크기 | FreeToken LRU | KTransformers prefill update | llama.cpp static |
| --- | ---: | ---: | ---: |
| Qwen3.6, expert pool의 37% | 16% miss | 41% miss | 62% miss |
| DSV4-Flash, expert pool의 11% | 39% miss | 59% miss | 89% miss |

Prefill에서 선택한 placement보다 decode 중 실제 route를 계속 따라가는 LRU가 temporal locality를 더 잘 잡았다. FreeToken의 $q^*$ 정책은 이 cache가 제거하지 못한 residual miss에 적용된다.

## 12. 작은 laptop부터 753B model까지

![FreeToken의 hardware별 coding-agent decode throughput]({{ '/assets/images/blog/freetoken-cross-hardware-decode.png' | relative_url }})

*RTX 4060 laptop부터 RTX 5090까지는 Qwen3.6 coding-agent workload, RTX PRO 6000은 GLM-5.2 math workload 결과다. 마지막 열은 별도의 frontier-scale demonstration이므로 앞 열과 model/workload가 다르다. 출처: FreeToken Figure 5.*

Figure 5의 다섯 consumer system에서는 같은 OpenCode workload로 Qwen3.6을 비교했다. FreeToken은 각 system의 가장 빠른 baseline보다 1.3–2.1배 높은 decode throughput을 보였다. 특히 RTX 4060 laptop 8 GB의 NVFP4 build는 39.3 tok/s를 기록했다.

이 수치는 논문이 인용한 production Codex trace의 median decode speed 33 tok/s보다 높다. 다만 서로 다른 모델, 서비스 환경, 측정 조건을 비교한 참고선이므로 “Codex보다 모델이 빠르다”는 품질 또는 end-to-end 비교로 해석하면 안 된다.

![각 hardware tier에서 실행 가능한 가장 강한 모델의 agentic decode speed]({{ '/assets/images/blog/freetoken-agentic-decode-speed.png' | relative_url }})

*각 hardware tier에서 그 장비가 수용할 수 있는 모델을 선택한 결과다. RTX 4060 laptop은 Qwen3.6-35B, RTX 5090 desktop은 DeepSeek-V4-Flash 284B, RTX PRO 6000은 GLM-5.2 753B를 사용하므로 세 막대를 같은 모델의 hardware scaling 결과로 읽으면 안 된다. 출처: FreeToken Figure 1(b) 일부.*

Figure 1(b)는 같은 모델의 scaling이 아니라 각 hardware tier가 실행할 수 있는 가장 강한 모델을 보여준다.

- RTX 4060 laptop 8 GB: Qwen3.6 NVFP4를 39.3 tok/s로 실행
- RTX 5090 desktop 32 GB: DeepSeek-V4-Flash 284B를 21.5 tok/s로 실행
- RTX PRO 6000 96 GB: GLM-5.2 753B를 14.9 tok/s로 실행

같은 RTX 5090 GPU를 사용한 server와 desktop 비교도 흥미롭다. Host memory bandwidth가 높은 server에서 dual-channel desktop으로 옮겼을 때 FreeToken의 decode rate는 약 4%만 감소했지만, llama.cpp는 server 성능의 80%만 유지했다. CPU, GPU, PCIe를 측정 bandwidth에 맞춰 나누는 정책이 host 구성 차이를 흡수한 결과로 해석할 수 있다.

GLM-5.2 753B의 checkpoint는 433 GB이며 RTX PRO 6000의 VRAM은 96 GB다. FreeToken은 전체 expert pool을 512 GiB host memory에 둔 구성에서 14.9 tok/s를 기록했다. llama.cpp의 7.3 tok/s보다 약 2배 높았고 mean TTFT는 7.5초 대 7.8초로 비슷했다.

## 13. 기존 시스템과 무엇이 다른가

FreeToken의 각 부품이 완전히 새로운 것은 아니다. Expert offloading, LRU cache, CPU-GPU hybrid execution, radix prefix cache는 모두 기존 연구와 시스템에 존재한다. 차이는 이들을 agentic edge workload에 맞춰 한 runtime에서 연결하는 방식에 있다.

| 접근 | 일반적인 방식 | FreeToken |
| --- | --- | --- |
| Expert placement | Load/prefill 때 static하게 결정 | Decode route를 따라가는 shared LRU |
| Cache miss | 모두 PCIe transfer 또는 모두 CPU | 측정 bandwidth로 transfer와 CPU를 동시 배분 |
| Prefill expert loading | On-demand 또는 compute와 직렬 | Full-layer double buffering |
| Prefix reuse | Full-attention KV 중심 | KV cache와 recurrent state checkpoint 결합 |
| VRAM split | Launch 때 고정 | Safe point에서 expert/KV cache 재구성 |
| Control path | Host가 dynamic schedule | GPU-resident control과 CUDA Graph capture |

FreeToken의 기여를 특정 cache algorithm 하나로만 보면 작아 보일 수 있다. 논문의 핵심은 **execution phase, semantic context structure, 실제 hardware bandwidth를 동시에 보고 cache와 compute placement를 조정하는 co-design**이다.

## 14. 한계와 읽을 때 주의할 점

### 아직 v1 시스템 논문이다

이 글이 다루는 것은 2026년 8월 공개된 arXiv v1에서 저자들이 보고한 측정 결과다. 아직 동료 심사나 독립 재현을 거친 결과로 읽어서는 안 된다. 넓은 hardware 범위를 다루지만 모든 조합을 같은 model과 workload로 평가한 것도 아니다. 특히 RTX PRO 6000의 GLM-5.2 결과는 Qwen3.6 coding-agent 실험과 분리된 demonstration이다.

### End-to-end wall-clock 비교는 아니다

Agent trajectory는 engine마다 달라질 수 있어 논문은 cross-engine 전체 wall-clock time을 직접 비교하지 않고 per-request mean decode throughput과 TTFT를 본다. 이 지표는 serving engine을 비교하는 데 유용하지만 task completion time 전체를 그대로 뜻하지 않는다.

### 큰 DRAM과 빠른 storage는 여전히 필요하다

VRAM 요구량을 줄였다고 model weight 자체가 사라지는 것은 아니다. 284B나 753B model을 실행하려면 전체 checkpoint를 담을 host memory와 storage가 필요하다. “개인용 GPU 한 장에서 실행”은 “GPU 한 장만 있으면 실행”과 다르다.

### `q*`는 측정값과 단순화된 bandwidth model에 의존한다

정책은 expert size와 bandwidth-bound execution을 전제로 branch 시간을 근사한다. 실제 desktop에서는 다른 process가 DRAM과 PCIe를 동시에 사용할 수 있다. FreeToken은 배포 시 $B_P$, $B_H$를 측정하지만, 논문은 급격한 runtime contention이 생겼을 때 이 값을 다시 추정하는지, 그렇다면 얼마나 자주 갱신하는지 설명하지 않는다. VRAM cache 재구성은 별도의 memory 관리 기능이므로 bandwidth 재측정과 같은 기능으로 읽으면 안 된다.

### Semantic anchor는 agent protocol에 의존한다

Special token으로 block 경계가 명확한 tool-calling protocol에서는 checkpoint가 잘 맞는다. 반대로 prompt를 임의 문자열 단위로 크게 재작성하거나 경계 token을 안정적으로 유지하지 않는 framework에서는 재사용 이득이 줄 수 있다.

### Free는 금전적 비용이 0이라는 뜻이 아니다

논문의 “FreeToken”과 hosted API 대비 “free” 표시는 token당 API 요금이 없다는 의미에 가깝다. Hardware 구입비, 전력, storage, setup과 운영 비용까지 0이 되는 것은 아니다.

## 15. 내가 이해한 핵심

FreeToken을 처음 보면 MoE weight를 CPU로 offloading하는 engine처럼 보인다. 하지만 내가 이해한 핵심은 GPU와 CPU 중 한쪽을 선택하는 것이 아니라 전체 memory hierarchy를 동시에 사용한다는 점이다.

이 논문이 보여주는 중요한 관점은 **희소성만으로는 로컬 추론이 빨라지지 않는다**는 것이다. MoE가 한 token의 계산량을 줄여도 전체 expert pool과 cache miss는 남는다. 결국 성능은 GPU FLOPs보다 data가 어디에 있고, 다음 layer와 token에서 누가 그 data를 읽을지에 의해 결정된다.

FreeToken은 단계별로 그 data movement를 다르게 다룬다.

```text
Prefill
거의 모든 expert가 필요하다
-> layer 전체를 미리 전송하고 GPU 계산과 overlap

Agent context edit
이전 state 일부가 무효가 된다
-> semantic boundary checkpoint에서 복원

Decode
일부 expert만 필요하다
-> LRU로 최근 working set을 VRAM에 유지
-> 남은 miss는 PCIe와 CPU에 bandwidth 비율로 분배

Runtime memory
KV cache와 expert cache 수요가 변한다
-> source of truth는 DRAM에 두고 VRAM cache를 재구성
```

가장 인상적인 부분은 $q^*$가 복잡한 predictor가 아니라 두 bandwidth의 비율로 나온다는 점이다. Miss를 더 잘 예측하는 것만으로는 PCIe ceiling을 넘기 어렵다. FreeToken은 miss가 생긴 뒤에도 CPU가 남는 DRAM bandwidth로 현재 token의 계산을 진행하게 해 PCIe와 CPU를 동시에 사용한다.

결국 이 논문의 메시지는 다음과 같다.

> Frontier model의 로컬 실행 가능성은 GPU memory 용량만으로 결정되지 않는다. GPU, CPU, DRAM, PCIe와 agent의 문맥 구조를 하나의 runtime이 얼마나 잘 조합하는지가 새로운 한계선을 만든다.

## 참고 자료

- [FreeToken paper](https://arxiv.org/abs/2608.16157)
- [FreeToken GitHub repository](https://github.com/FlashML-org/FreeToken)
- [FreeToken project page](https://flashml.ai)
