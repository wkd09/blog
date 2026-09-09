---
title: "Disaggregated Serving 정리: Prefill과 Decode를 분리하는 LLM 추론 아키텍처"
date: 2026-09-09 00:00:00 +0900
last_modified_at: 2026-09-09 00:00:00 +0900
summary: "LLM 추론의 compute-bound prefill과 memory-bandwidth-bound decode를 별도 worker로 분리하는 이유, KV cache 전달과 scheduling이 만드는 이점과 비용, 실제 도입 기준을 정리한다."
categories:
  - engineering
tags:
  - AI
  - LLM
  - Inference
  - Serving
  - Disaggregated Serving
  - Prefill
  - Decode
  - KV Cache
  - NVIDIA Dynamo
source: "NVIDIA Disaggregated Serving Glossary, Kimi Deep Research"
---

# Disaggregated Serving: Prefill과 Decode를 왜 분리할까?

LLM inference server는 하나의 요청을 받아 prompt를 읽고, 첫 token을 만든 뒤, 나머지 token을 차례대로 생성한다. 겉으로는 하나의 연속된 작업처럼 보이지만 GPU 입장에서는 성격이 다른 두 workload다.

- **Prefill**은 prompt token을 병렬로 처리하며 큰 matrix multiplication을 수행한다.
- **Decode**는 이전 결과를 바탕으로 한 번에 한 token씩 생성하며 model weight와 KV cache를 계속 읽는다.

전통적인 serving에서는 같은 GPU worker가 두 단계를 모두 수행한다. 구현은 단순하지만 compute를 많이 쓰는 prefill과 memory bandwidth를 많이 쓰는 decode가 같은 자원과 scheduler를 공유한다. 긴 prompt 하나가 들어오면 이미 답을 생성 중인 요청의 token 간격이 늘어날 수 있고, decode 요청을 많이 batch하면 새 요청의 첫 token이 늦어질 수 있다.

[NVIDIA의 설명](https://www.nvidia.com/en-us/glossary/disaggregated-serving/)처럼 **Disaggregated Serving**은 두 단계를 전용 GPU 또는 worker pool로 분리하고, 각 단계에 맞는 자원과 scaling 정책을 적용하는 아키텍처다.

```text
Aggregated serving

Request -> [ GPU worker: Prefill + Decode ] -> Response


Disaggregated serving

Request -> Router -> [ Prefill worker ]
                           │
                           │ KV cache transfer
                           ▼
                       [ Decode worker ] -> Response
```

한 줄로 요약하면 다음과 같다.

> Disaggregated Serving은 model을 쪼개는 기술이 아니라, 한 요청의 서로 다른 실행 단계를 분리해 각각 최적화하는 serving 아키텍처다.

## 1. LLM 추론은 두 종류의 workload다

### Prefill: prompt를 한꺼번에 처리하는 단계

Prefill에서는 입력 prompt 전체를 model에 통과시킨다. 각 layer는 여러 token을 병렬로 계산하고, 이후 생성에 필요한 key와 value를 KV cache에 저장한다.

Prompt 길이를 $L$, model layer 수를 $N$이라고 하면 attention 계산량은 단순히 token 하나를 처리하는 것보다 훨씬 크다. 대신 충분한 token을 묶어 큰 matrix multiplication을 만들 수 있으므로 GPU의 연산 장치를 효율적으로 사용한다. 이 단계는 일반적으로 **compute-bound**에 가깝다.

사용자가 체감하는 대표 지표는 `TTFT(Time To First Token)`다.

```text
request ─────────────── first token
         <--- TTFT --->
         prefill
```

긴 문서, 큰 codebase, 많은 tool definition을 입력할수록 prefill 시간이 길어진다.

### Decode: token을 하나씩 생성하는 단계

Decode에서는 방금 생성한 token을 다시 입력으로 넣어 다음 token을 만든다. 한 step에서 처리하는 새 token 수가 작아 prefill만큼 큰 matrix multiplication을 만들기 어렵다. 반면 매 step마다 model weight와 지금까지 쌓인 KV cache를 읽어야 한다.

그래서 decode는 보통 FLOPS보다 **memory capacity와 memory bandwidth**의 영향을 크게 받는다. 대표 지표는 token 사이의 간격인 `ITL(Inter-Token Latency)` 또는 `TPOT(Time Per Output Token)`다.

```text
first token ── token 2 ── token 3 ── token 4
             <- ITL ->   <- ITL ->
                   decode
```

두 단계의 차이를 표로 정리하면 다음과 같다.

| 구분 | Prefill | Decode |
| --- | --- | --- |
| 처리 단위 | Prompt token 다수 | Step마다 새 token 소수 |
| 대표 병목 | GPU compute | Memory bandwidth·capacity |
| 주요 상태 | KV cache 생성 | KV cache 읽기·증가 |
| 사용자 지표 | TTFT | ITL, TPOT |
| 유리한 운영 | 큰 batch, 높은 연산량 | 지속적 batching, 빠른 memory 접근 |

## 2. 같은 GPU에 함께 두면 왜 문제가 생길까?

Aggregated serving에서는 한 worker가 prefill과 decode request를 같은 queue에서 처리한다. 이때 두 workload가 서로 간섭한다.

예를 들어 긴 prompt의 prefill을 한 번에 실행하면 GPU를 오래 점유해 진행 중인 decode step이 기다린다. 이미 streaming 중인 사용자는 답변이 중간중간 끊기는 현상을 체감할 수 있다. 반대로 decode batch를 자주 우선 처리하면 새 request의 prefill 시작이 밀려 TTFT가 늘어난다.

Scheduler가 chunked prefill, continuous batching, priority policy를 사용하면 이 간섭을 줄일 수 있다. 하지만 두 workload가 같은 GPU fleet에 묶여 있다는 구조는 남는다.

```text
하나의 GPU pool에서 생기는 긴장

Prefill을 크게 실행 -> compute 효율 증가, decode ITL 악화 가능
Decode를 자주 실행  -> streaming 안정, prefill TTFT·throughput 악화 가능
```

Scaling도 비효율적일 수 있다. 입력은 길고 출력은 짧은 workload에서는 prefill capacity가 더 필요하다. 짧은 질문에 긴 답을 만드는 workload에서는 decode capacity가 더 필요하다. 하나의 pool만 사용하면 부족한 단계 하나를 늘리기 위해 다른 단계의 자원까지 같은 비율로 추가해야 한다.

Disaggregated serving은 이 결합을 끊는다.

## 3. 요청은 어떻게 Prefill worker에서 Decode worker로 이동할까?

분리된 시스템의 request lifecycle은 다음과 같다.

```text
1. Router가 요청을 받을 prefill worker를 선택한다.
2. Prefill worker가 prompt를 처리하고 KV cache를 만든다.
3. Scheduler가 decode worker를 선택한다.
4. KV cache를 prefill worker에서 decode worker로 전달한다.
5. Decode worker가 cache를 이어받아 token을 생성한다.
6. 새 token을 streaming하고 KV cache를 계속 확장한다.
```

여기서 핵심은 model output이 아니라 **KV cache의 소유권이 worker 사이에서 이동한다는 점**이다. Prefill이 만든 상태 없이 decode를 시작하면 prompt 전체를 다시 계산해야 하므로 분리의 이점이 사라진다.

KV cache 크기는 대략 다음과 같이 계산할 수 있다.

$$
KV\ cache\ bytes \approx 2 \times N_{layers} \times N_{kv\ heads} \times D_{head} \times L \times B
$$

- 앞의 $2$는 key와 value를 뜻한다.
- $L$은 현재 sequence length다.
- $B$는 element 하나의 byte 수다.

예를 들어 80개 layer, 8개 KV head, head dimension 128, BF16을 사용하는 GQA model이라면 sequence 하나의 KV cache는 token당 약 320 KiB다. 32K context에서는 약 10 GiB가 된다. Batch의 여러 sequence를 옮기면 transfer volume은 더 커진다.

따라서 disaggregation의 성패는 다음 부등식으로 생각할 수 있다.

$$
분리로\ 줄인\ 대기시간과\ 낭비 > KV\ transfer와\ orchestration\ 비용
$$

고속 interconnect, GPU-to-GPU 전송, 비동기 copy가 중요한 이유가 여기에 있다. 느린 network에서 큰 cache를 매번 복사하면 새 병목만 만들 수 있다.

## 4. 분리하면서 얻는 세 가지 자유도

### Prefill과 Decode를 독립적으로 확장할 수 있다

Traffic 특성에 맞춰 두 pool의 크기를 따로 조절할 수 있다.

```text
Long input / short output  -> Prefill worker 비중 증가
Short input / long output  -> Decode worker 비중 증가
Traffic 급증               -> Queue와 SLO를 보고 각 pool을 독립 scaling
```

단순 request 수만 보고 autoscaling해서는 부족하다. Prefill queue에는 input token 수, decode queue에는 active sequence 수와 예상 output token 수가 더 직접적인 load signal이 된다.

### 단계별 SLO를 따로 관리할 수 있다

TTFT와 ITL은 서로 다른 사용자 경험을 나타낸다. 두 단계를 분리하면 prefill scheduler는 TTFT를, decode scheduler는 안정적인 token generation을 중심으로 정책을 설계할 수 있다.

예를 들어 interactive chat은 ITL jitter가 작아야 자연스럽다. Batch summarization은 개별 token 간격보다 전체 throughput이 중요하다. 같은 model을 사용해도 endpoint나 priority에 따라 서로 다른 pool과 scheduling policy를 적용할 수 있다.

### 서로 다른 hardware 구성을 선택할 수 있다

Prefill에는 높은 compute throughput을 가진 accelerator를, decode에는 충분한 memory capacity와 bandwidth를 가진 accelerator를 배치할 수 있다. NVIDIA도 disaggregated serving의 장점으로 이러한 heterogeneous hardware 배치를 설명한다.

다만 처음부터 반드시 서로 다른 GPU SKU를 사용해야 하는 것은 아니다. 동일한 GPU를 사용하더라도 역할별 pool을 분리하면 scheduling 간섭을 줄이고 독립 scaling을 할 수 있다. 운영 복잡도를 낮추기 위해 같은 hardware로 시작한 뒤 profile 결과에 따라 이기종 구성을 검토하는 방법도 현실적이다.

## 5. 가장 어려운 부분은 GPU가 아니라 Data Plane이다

그림만 보면 GPU pool 두 개와 router 하나면 충분해 보인다. 실제 production system에서는 다음 구성 요소가 함께 필요하다.

### KV cache transfer

Cache를 host memory에 내렸다가 다시 올리는지, RDMA나 GPU direct path로 전달하는지에 따라 latency가 달라진다. Transfer와 computation을 겹칠 수 있어야 하고, 실패했을 때 중간 상태를 폐기하거나 다시 prefill하는 정책도 필요하다.

### KV cache-aware routing

Decode worker의 free memory만 보고 routing하면 부족하다. 어느 worker가 어떤 cache block을 보유하고 있는지 알아야 불필요한 이동과 재계산을 줄일 수 있다. Multi-turn chat이나 shared prefix cache까지 사용한다면 cache locality가 더욱 중요하다.

### Rate matching과 backpressure

Prefill pool이 decode pool보다 빠르면 완성된 KV cache가 대기 queue에 쌓인다. 이 cache는 GPU memory를 점유하고, 오래 기다린 request의 TTFT 이후 지연도 커진다. Decode가 더 빠르면 worker가 새 cache를 기다리며 idle 상태가 된다.

```text
Prefill rate > Decode 수용량
               ↓
KV cache 대기 증가 -> memory pressure -> admission control 필요
```

두 단계의 처리율을 맞추고, queue가 임계값을 넘으면 새 request를 늦추거나 다른 replica로 보내는 backpressure가 필요하다.

### 장애와 lifecycle 관리

Prefill이 끝난 직후 decode worker가 실패하면 어느 단계부터 재시도할지 결정해야 한다. Cache transfer가 부분적으로 끝난 상태도 처리해야 한다. Worker health, cache metadata, request state를 함께 관리하므로 단일 worker serving보다 failure mode가 많다.

[NVIDIA Dynamo](https://www.nvidia.com/en-us/ai-data-science/products/dynamo/) 같은 distributed inference framework가 routing, KV cache transfer, memory management를 하나의 문제로 다루는 이유다. 단순 kernel 최적화만으로는 disaggregated serving을 완성할 수 없다.

## 6. 어떤 workload에서 효과가 큰가?

Disaggregated serving은 다음 조건에서 검토할 가치가 크다.

- 긴 context를 입력하는 code generation, document analysis, deep research workload
- Input/output 길이 분포가 넓어 prefill과 decode 부하 비율이 자주 달라지는 서비스
- 동시 request가 많아 continuous batching과 독립 scaling의 이점이 큰 환경
- TTFT와 ITL SLO를 동시에 안정적으로 관리해야 하는 interactive service
- 여러 GPU node를 연결할 충분히 빠른 network와 관측 가능한 serving stack을 갖춘 환경

반대로 항상 더 좋은 선택은 아니다.

- 작은 model이나 낮은 traffic에서는 worker 분리와 network hop의 고정 비용이 더 클 수 있다.
- Input/output 길이가 일정하고 잘 조정된 aggregated server로 SLO를 만족한다면 복잡도를 늘릴 이유가 약하다.
- KV cache를 빠르게 전달할 수 없는 network에서는 긴 context일수록 손해가 커질 수 있다.
- Prefill과 decode pool 중 한쪽이 자주 idle하면 독립 scaling의 이점보다 fragmentation이 커진다.

Kimi의 [Disaggregated Inference 정리](https://www.kimi.com/preview/19a6a7a7-4d62-84cd-8000-057fb1def79b)는 비용 절감, GPU utilization, transfer latency 개선 수치를 여러 사례에서 모아 소개한다. 이런 수치는 가능성을 보여주지만 model, sequence 길이, concurrency, hardware, network, SLO가 달라지면 그대로 재현되지 않는다. NVIDIA가 제시하는 `up to 15x` throughput이나 ROI 역시 특정 benchmark와 system configuration의 상한으로 읽어야 한다.

따라서 “분리하면 몇 % 빨라지는가?”보다 먼저 다음을 측정해야 한다.

## 7. 도입 전 측정해야 할 지표

| 영역 | 확인할 지표 | 이유 |
| --- | --- | --- |
| Traffic | Input/output token 분포, 동시 요청 수 | Prefill·decode capacity 비율 결정 |
| Latency | TTFT p50/p95/p99, TPOT·ITL p95/p99 | 단계별 SLO와 interference 확인 |
| Throughput | Prefill tokens/s, decode tokens/s | 두 pool의 rate matching |
| Cache | Request당 KV cache 크기, transfer bytes/s | Network와 memory 요구량 계산 |
| Resource | GPU compute, HBM bandwidth, memory 사용률 | 실제 병목이 compute인지 memory인지 확인 |
| Queue | Prefill queue time, decode admission delay | Pool 부족과 backpressure 감지 |
| Cost | GPU-hour당 처리 token, request당 비용 | 성능 향상이 비용 절감으로 이어지는지 확인 |

도입 순서는 다음처럼 잡을 수 있다.

1. Aggregated serving에서 prefill과 decode 시간을 분리해 측정한다.
2. 긴 prefill이 decode ITL을 흔드는지 trace로 확인한다.
3. 실제 model과 context 길이로 KV cache 크기와 전송 시간을 계산한다.
4. 두 pool의 처리율을 benchmark해 필요한 replica 비율을 정한다.
5. 평균이 아니라 p95/p99 latency와 비용을 함께 비교한다.
6. Traffic mix가 바뀌었을 때 autoscaling과 failure recovery를 검증한다.

특히 비교 실험에서는 같은 model, quantization, request trace, SLO를 사용해야 한다. Disaggregated 구성이 더 많은 request를 처리하더라도 TTFT나 ITL 조건을 완화했다면 공정한 비교가 아니다.

## 8. 다른 분산 기술과 무엇이 다른가?

이름 때문에 tensor parallelism이나 pipeline parallelism과 혼동하기 쉽다.

| 기술 | 무엇을 나누는가? | 주된 목적 |
| --- | --- | --- |
| Tensor Parallelism | 한 layer의 tensor 계산 | 큰 model의 병렬 계산 |
| Pipeline Parallelism | Model layer | 여러 device에 model 배치 |
| Data Parallel Serving | Request·model replica | 처리량과 가용성 증가 |
| Disaggregated Serving | Prefill·decode 실행 단계 | 단계별 병목 최적화와 독립 scaling |

서로 대체 관계도 아니다. Prefill worker 하나가 여러 GPU의 tensor parallel group일 수 있고, decode worker도 별도의 parallel group일 수 있다.

```text
Request
  │
  ▼
Prefill replica (TP=4 GPUs)
  │ KV cache
  ▼
Decode replica  (TP=2 GPUs)
```

즉 model parallelism은 “한 단계의 model 계산을 어떻게 여러 GPU에 배치할까?”를 다루고, disaggregated serving은 “서로 다른 단계를 어떤 worker에 맡길까?”를 다룬다.

## 9. 정리

Disaggregated serving이 해결하려는 문제는 분명하다.

```text
Prefill = compute-heavy, TTFT 중심
Decode  = memory-heavy, ITL 중심
```

하나의 GPU pool에 두 workload를 함께 두면 scheduling 간섭과 결합된 scaling이 발생한다. 두 단계를 분리하면 단계별 hardware, batching, SLO, replica 수를 독립적으로 조정할 수 있다.

그러나 그 대가로 KV cache transfer, cache-aware routing, rate matching, backpressure, failure recovery라는 distributed system 문제가 생긴다. 결국 핵심 판단 기준은 아키텍처의 새로움이나 특정 최대 성능 수치가 아니다.

> 실제 workload에서 단계 분리로 얻는 이득이 KV cache 이동과 운영 복잡도의 비용보다 큰가?

긴 context와 높은 concurrency가 일반화될수록 이 질문은 더 중요해진다. Disaggregated serving은 모든 LLM server의 기본값이라기보다, aggregated serving의 간섭과 scaling 한계가 측정으로 드러났을 때 사용할 수 있는 강력한 다음 단계다.

## 참고 자료

- [NVIDIA, What Is Disaggregated Serving?](https://www.nvidia.com/en-us/glossary/disaggregated-serving/)
- [Kimi, Disaggregated Inference: The Architectural Revolution in AI Model Serving](https://www.kimi.com/preview/19a6a7a7-4d62-84cd-8000-057fb1def79b)
- [KV Cache의 구조와 memory 계산]({% post_url 2026-06-21-large-kv-cache-long-context %})
- [vLLM과 PagedAttention 정리]({% post_url 2026-05-29-vllm %})
