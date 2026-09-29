---
title: "HySparse 정리: Full Attention을 오라클로 활용하는 하이브리드 Sparse Attention"
date: 2026-09-29 00:00:00 +0900
summary: "HySparse가 full attention에서 중요한 block과 KV cache를 가져와 sparse layer에 재사용하는 방식, 성능 결과와 남은 과제를 정리한다."
categories:
  - research
tags:
  - AI
  - LLM
  - Transformer
  - Attention
  - Sparse Attention
  - KV Cache
  - Long Context
source: "Gao et al., HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing, arXiv:2602.03560v1"
---

긴 문맥을 처리하는 Transformer의 부담은 attention 계산량에만 있지 않다. 각 layer가 긴 sequence의 Key와 Value를 보관해야 하므로, decode 중 GPU 메모리에 쌓이는 KV cache도 커진다. Sparse attention은 일부 token만 골라 계산량을 낮출 수 있지만, 중요한 token을 어떻게 고를지와 KV cache를 어떻게 줄일지가 별개의 난제로 남는다.

**HySparse**는 이 두 문제를 layer 사이의 협력으로 다룬다. 먼저 실행되는 full attention이 중요한 token block을 고르고 KV cache를 만든다. 뒤따르는 여러 sparse attention layer는 선택된 block과 그 KV를 재사용한다. 여기에 짧은 거리 정보를 담당하는 sliding-window attention(SWA) 경로를 더한다.

이 글은 [HySparse 논문](https://arxiv.org/abs/2602.03560)의 구조와 실험을 정리하고, 실제 시스템에 적용할 때 확인할 점을 살펴본다.

## Sparse Attention은 무엇을 줄이고 무엇을 남기는가

Full attention에서 query는 앞선 모든 key와 value를 대상으로 계산한다. sequence 길이를 `n`이라 하면 attention 계산량은 대략 `O(n²)`에 비례한다. Sparse attention은 query마다 관련성이 높은 일부 token이나 block만 선택해 이 연산을 줄인다.

하지만 동적 sparse attention이 계산량을 줄여도 전체 KV cache를 그대로 보관하는 경우가 많다. 생성 도중 중요도가 달라질 수 있어, 미리 버린 KV를 되살리기 어렵기 때문이다. 반대로 고정된 window만 남기면 메모리는 절약하지만 멀리 떨어진 중요한 정보에 접근하지 못할 수 있다.

HySparse는 두 문제를 한 구조에서 연결한다.

1. **선택의 정확도:** proxy나 별도 indexer 대신 앞선 full attention의 실제 attention 점수로 중요 block을 선택한다.
2. **KV 메모리:** sparse attention branch는 직전 full layer가 만든 KV cache를 공유해 layer마다 전체 KV를 추가 저장하지 않는다.
3. **근거리 표현력:** sparse global branch와 별개로 SWA branch는 자체 KV cache를 유지한다.

## 한 번의 Full Attention 뒤에 여러 Sparse Layer

HySparse는 Transformer layer를 `full attention 1개 → sparse attention N개` 형태의 묶음으로 구성한다. 각 묶음 안에서 sparse layer들은 바로 앞 full attention layer가 제공한 block index와 K/V를 활용한다.

```text
입력
  ↓
Full Attention ──→ 중요 block index와 K/V cache 생성
  ↓                         │
Sparse Attention 1 ←────────┤ 선택 block의 KV 재사용
  ↓                         │
Sparse Attention 2 ←────────┤ 같은 index와 KV 재사용
  ↓                         │
...                         │
  ↓                         │
MoE / FFN                   │
```

논문 설정은 7B dense 모델에서 full:sparse 비율 1:3, 80B-A3B MoE에서 1:11이다. 따라서 80B 모델의 49개 layer 중 full attention을 쓰는 layer는 5개뿐이다. 각 하이브리드 묶음의 마지막 layer는 full attention으로 두어 최종 global aggregation을 보장한다.

![HySparse의 Full Attention과 뒤따르는 Sparse Attention layer 구조 및 KV와 block index 재사용 경로](/assets/images/blog/hysparse-architecture.png)

*HySparse는 앞선 full attention에서 얻은 K/V와 Top-K block index를 다음 sparse layer들에 전달한다. Sparse layer 안에서는 공유 KV를 쓰는 block sparse branch와 자체 KV를 유지하는 SWA branch를 gate로 합친다. 출처: HySparse 논문 Figure 1.*

### Full Attention이 다음 Layer의 선택기를 겸한다

Full attention은 원래 현재 layer의 출력만 계산하는 것이 아니라, 이후 sparse layer에서 쓸 중요 block도 제공한다. 논문은 attention score 전체 행렬을 저장하는 대신 block 단위 점수를 만든다. FlashAttention의 online softmax 계산 중 얻는 중간값을 활용해 block별 중요도를 만들고, Top-K로 다음 sparse layer에서 참고할 block을 정한다.

기본 설정은 block 크기 64 token, Top-K 1024 token이다. 따라서 선택은 최대 16개 block에 해당한다. GQA에서는 query head 그룹 안에서 점수를 모아 같은 그룹이 index를 공유하도록 한다. 이는 index 관리와 sparse kernel 실행을 단순화한다.

여기서 논문이 말하는 “oracle”은 정답 라벨을 아는 신비한 선택기가 아니다. **바로 앞 layer에서 full attention을 이미 계산했으므로, 그 실제 attention에서 추출한 정보를 선택에 사용한다**는 뜻이다. 별도의 proxy가 중요도를 추정할 때 생기는 오차를 피한다.

### Sparse Global 경로와 Local SWA 경로

각 sparse layer는 두 갈래의 attention 결과를 결합한다.

```text
                         ┌─ Block Sparse Attention ─ 선택된 먼 block의 KV 공유 ─┐
입력 ─ Q projection ─────┤                                                     ├─ gate로 결합
                         └─ Sliding Window Attention ─ 독립된 작은 KV cache ──┘
```

Block sparse branch는 선택된 global context를 효율적으로 가져온다. SWA branch는 최근 token 주변의 연속적인 단서를 따로 모델링한다. 두 출력에는 sigmoid gate를 적용해 각 위치에서 local과 global 정보를 섞는다.

이 설계에서 중요한 세부점은 **KV 공유를 모든 branch에 강제하지 않는 것**이다. Sparse global branch는 이전 full layer의 KV를 공유하지만, SWA branch는 자체 KV를 둔다. 논문 ablation에서는 SWA까지 공유 KV를 쓰면 점수가 눈에 띄게 낮아졌다. 저자들은 global retrieval용으로 만들어진 KV가 짧은 거리 표현에 필요한 정보를 충분히 담지 못할 수 있다고 해석한다.

## 실험 결과

저자들은 7B dense 모델과 80B-A3B MoE 모델을 학습해 Full Attention, Hybrid SWA, HySparse를 비교했다. Hybrid SWA는 같은 hybrid 비율로 full attention과 sliding-window attention을 섞고, HySparse는 sparse global 경로를 추가한 구조다. 아래는 첨부한 논문 표와 본문 결과 중 전체 점수 및 대표 결과를 요약한 것이다.

![7B Dense와 80B MoE 모델의 일반 추론 벤치마크 결과 비교](/assets/images/blog/hysparse-benchmark-results.png)

*일반, 수학, 코드, 중국어 benchmark 비교. HySparse는 대부분 항목에서 Full Attention 또는 Hybrid SWA와 비슷하거나 높은 성능을 보이며, 모든 개별 benchmark에서 우세한 것은 아니다. 출처: HySparse 논문 Table 2.*

| 평가 | 모델 | Full Attention | Hybrid SWA | HySparse |
|---|---:|---:|---:|---:|
| 일반·추론 종합 경향 | 7B | 기준 | 대체로 비슷하거나 하락 | 대부분 유지 또는 개선 |
| RULER 16K 전체 | 7B | 93.0 | 91.6 | **94.1** |
| RULER 32K 전체 | 7B | 88.2 | 84.2 | **89.3** |
| RULER 16K 전체 | 80B MoE | **93.6** | 72.7 | 90.6 |
| RULER 32K 전체 | 80B MoE | 82.1 | 69.5 | **87.4** |
| GSM8K | 80B MoE | 53.8 | 45.3 | **54.1** |
| C-Eval | 80B MoE | 64.6 | 58.8 | **65.0** |

![7B Dense와 80B MoE 모델의 16K 및 32K RULER 결과 비교](/assets/images/blog/hysparse-ruler-results.png)

*RULER의 개별 과제와 종합 점수. 80B MoE에서 HySparse의 종합 점수는 16K에서 90.6, 32K에서 87.4다. 출처: HySparse 논문 Table 3.*

80B MoE의 설정은 full:sparse 1:11이며, 49개 layer 가운데 full attention layer가 5개다. 이 조건에서 HySparse는 32K RULER 전체 점수 87.4로 Full Attention의 82.1을 넘었고, Hybrid SWA의 69.5보다 크게 높았다. 논문은 이 구조가 Hybrid SWA와 비교해 KV cache 저장량을 거의 10배 줄인다고 보고한다.

다만 모든 벤치마크에서 HySparse가 항상 Full Attention보다 높은 것은 아니다. 80B 일반 평가에서 MMLU-Pro, DROP, ARC-Challenge 등 일부 점수는 Full Attention보다 조금 낮았다. RULER 16K에서도 80B의 종합 점수는 Full Attention이 더 높다. 결과를 “희소화가 언제나 정확도를 높인다”로 읽기보다, **더 적은 full layer로도 긴 문맥 성능을 유지하는 유력한 절충안**으로 보는 편이 정확하다.

7B ablation은 두 가지 설계 선택을 뒷받침한다. 첫째, SWA branch를 제거하면 GSM8K가 37.7에서 29.7로 내려가는 등 여러 항목이 악화됐다. 둘째, sparse branch와 SWA branch 모두 공유 KV를 쓰는 구성보다 sparse branch만 공유하는 구성이 더 나았다. local pathway는 별도 표현을 유지할 필요가 있다는 결과다.

## Serving 관점에서의 의미

HySparse의 KV 절감은 sparse attention이 계산 대상만 줄이는 방식에서 한 걸음 더 나아간다. 한 묶음의 여러 sparse layer가 KV를 공유하므로, sparse layer마다 긴 context KV를 복제해 보관할 필요가 없다. serving에서는 같은 GPU 메모리로 더 많은 요청을 올리거나, 더 긴 context를 수용할 가능성이 생긴다.

논문은 추가 방향으로 full attention layer의 KV를 외부 메모리로 내보내고, GPU에는 선택된 sparse KV를 유지하는 offloading 아이디어도 언급한다. 이는 제안된 추론 시스템의 측정 결과라기보다 향후 활용 가능성이다. 메모리 절감이 실제 처리량이나 지연 개선으로 이어지는지는 KV 이동 비용, kernel 효율, batch 크기, context 길이를 포함해 별도로 검증해야 한다.

또한 HySparse는 inference-only patch가 아니라 sparse 구조를 포함해 학습한 모델이다. 논문 7B 모델은 1T token 학습 후 200B token을 32K 길이로 추가 학습했고, 80B MoE 모델은 32K 길이로 500B token을 학습했다. 기존 체크포인트에 sparse mask만 덧씌우는 것과 같은 조건으로 볼 수 없다.

## 한계와 확인할 점

- **완전한 선형 복잡도는 아니다.** Full attention layer를 없애지 않으며, 저자들도 full attention 및 선택 단계의 비용을 완전히 제거하는 것은 여전히 어렵다고 설명한다.
- **선택은 이전 layer 신호에 의존한다.** 앞선 full attention의 중요도가 다음 여러 layer에서도 유효하다는 가정이 필요하다. context와 표현이 변하는 모든 모델에서 선택 품질이 보장되는 것은 아니다.
- **학습 비용이 크다.** 보고된 결과는 대규모 추가 학습을 포함한다. 사전학습 비용과 효율 개선의 손익을 함께 따져야 한다.
- **평가 범위가 제한적이다.** 두 모델 규모와 논문에 포함된 benchmark 결과만으로 다양한 모델 계열, 긴 생성, 실제 동시 serving 부하까지 일반화할 수는 없다.
- **계산량 감소가 곧 latency 개선은 아니다.** Block sparse kernel과 index 처리, KV 이동이 하드웨어에서 잘 구현되어야 이론적 절감이 실제 속도와 처리량으로 이어진다.

## 정리

HySparse의 핵심은 sparse attention을 고립된 layer의 기법으로 보지 않고, **full attention이 다음 sparse layer들을 위한 정보 제공자 역할도 하도록 설계한 것**이다. 실제 attention에서 뽑은 중요 block으로 proxy 선택기를 대체하고, 만들어 둔 KV를 여러 sparse layer가 공유해 계산과 cache 메모리를 함께 줄인다. 그리고 local SWA에는 자체 KV를 남겨 근거리 표현을 보완한다.

특히 80B MoE에서 full attention을 49개 중 5개 layer만 쓰면서도 긴 문맥 benchmark에서 경쟁력 있는 점수를 낸 결과가 눈에 띈다. 다음으로 확인해야 할 것은 이 학습된 구조가 실제 serving의 latency, throughput, GPU 메모리에서 어떤 수치로 이어지는지다.

## 참고 자료

- Gao et al., [HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing](https://arxiv.org/abs/2602.03560), arXiv:2602.03560v1, 2026.
