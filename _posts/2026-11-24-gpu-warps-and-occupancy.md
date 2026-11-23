---
title: "03 GPU 실행 모델과 점유율"
date: 2026-11-24 00:00:00 +0900
summary: "NVIDIA GPU가 스레드를 워프로 묶는 이유, 분기가 실행 레인에 미치는 영향, 점유율과 명령어 독립성이 메모리 지연을 숨기는 방법을 살펴본다."
categories:
  - study
tags:
  - GPU
  - CUDA
  - SIMT
  - Occupancy
  - Computer Architecture
---

지난 글에서는 워프 하나가 메모리를 기다릴 때 GPU가 준비된 다른 워프를 실행한다고 했다. 그런데 워프 32개가 SM에 올라와 있어도 그중 실행할 수 있는 워프가 하나도 없을 수 있다. 반대로 같은 워프에 있는 32개 스레드가 모두 일을 갖고 있어도, 다음 명령을 실행할 때는 그중 16개만 참여할 수 있다. 두 상황을 구별하면 GPU가 어디서 시간을 잃는지 읽기 쉬워진다. [이전 글 · 02 GPU 대역폭]({{ site.baseurl }}{% post_url 2026-11-19-gpu-memory-bandwidth %})

## 스레드 32개를 왜 워프로 묶을까?

CUDA 커널에서 픽셀 32개의 밝기를 바꾼다고 하자. 스레드마다 담당 픽셀과 레지스터에 담긴 값은 다르지만, 할 일은 같다. 각 스레드가 명령을 따로 가져와 해독하는 대신 SM(Streaming Multiprocessor)은 스레드를 묶고, 워프 스케줄러는 묶음 안에서 준비된 스레드에 같은 명령을 발행한다. 덕분에 여러 스레드가 명령어 제어 비용을 나눠 쓴다.

![96개 스레드는 세 워프를 모두 채운다. 100개 스레드는 네 번째 워프의 32개 레인 중 4개를 채운다.]({{ '/assets/images/blog/gpu-warps-occupancy/block-warps.svg' | relative_url }})

NVIDIA CUDA에서 **워프(warp)는 같은 블록의 연속된 스레드 32개를 묶는 논리적 실행 단위**다. 그림에서 96개 스레드는 세 워프를 모두 채운다. 100개라면 마지막 4개가 네 번째 워프를 차지하고 나머지 28개 레인은 비어 있다. 워프의 크기는 스레드 수를 나타낸다. 명령 하나가 32개 스레드에서 완료되기까지 걸리는 시간은 연산 종류와 GPU의 실행 자원에 따라 달라진다. 32는 NVIDIA가 선택한 크기다. [그림 2.8 · Arpan Pathak](https://arpanpathak.github.io/gpu-parallel-book/chapters/part1_foundations/ch02_gpu_hardware_first_principles.html#24-blocks-grids-and-warps) · [CUDA Programming Guide · Hardware Multithreading](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)

## 같은 워프 안에서 분기가 갈리면

1차원 스레드 블록에서 모든 레인이 찬 워프 하나를 생각해 보자.

~~~cpp
if (threadIdx.x % 2 == 0) {
    a = a * 2;
} else {
    a = a + 5;
}
~~~

짝수 스레드 16개는 곱셈을, 홀수 스레드 16개는 덧셈을 선택한다. 이처럼 워프 내부에서 실행 경로가 갈리는 현상이 **분기 발산(branch divergence)**이다. 곱셈 경로의 명령을 발행할 때는 짝수 레인이 참여하고, 덧셈 경로에서는 홀수 레인이 참여한다.

![8개 레인으로 줄여 그린 예시. x가 2보다 큰 레인은 if에서, 나머지 레인은 else에서 활성화되고 분기 뒤에는 다시 모두 활성화된다.]({{ '/assets/images/blog/gpu-warps-occupancy/branch-divergence.svg' | relative_url }})

그림에서는 조건을 x가 2보다 큰지로 바꿔 8개 레인만 보여준다. 색칠된 칸을 따라가면 두 경로에서 실행에 참여하는 레인이 어떻게 바뀌는지 보인다. 본문의 16 대 16 예시에서는 각 경로에 레인의 절반이 참여하지만, 그것만으로 커널이 두 배 느려진다고 계산할 수는 없다. 경로마다 명령 수가 다를 수 있고, 짧은 분기는 컴파일러가 조건부 실행(predication)으로 바꾸기도 한다. [그림 2.7 · Arpan Pathak](https://arpanpathak.github.io/gpu-parallel-book/chapters/part1_foundations/ch02_gpu_hardware_first_principles.html#233-divergence) · [CUDA Best Practices Guide · Branching and Divergence](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#branching-and-divergence)

| 상황 | 워프 내부 발산 |
| --- | --- |
| 워프 A 전체가 if, 워프 B 전체가 else를 선택 | 없음 |
| 같은 워프의 16개 스레드가 if, 16개가 else를 선택 | 있음 |

NVIDIA Volta 이후의 **Independent Thread Scheduling**은 스레드마다 프로그램 카운터와 호출 스택을 관리한다. 스케줄러는 같은 워프에서 다음 명령을 함께 실행할 스레드를 묶을 수 있다. 같은 워프라는 이유만으로 서로의 작업이 끝났다고 가정하면 동기화 오류가 생길 수 있다. [CUDA Programming Guide · SIMT Architecture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#simt-architecture)

## SIMD와 SIMT는 어디가 다를까?

둘 다 한 명령으로 여러 데이터에 연산을 적용한다. 차이는 코드를 작성할 때 보이는 단위다. **SIMD**에서는 한 CPU 스레드의 AVX-512 명령이 512비트 벡터 레지스터에 담긴 FP32 값 16개를 한 번에 더할 수 있다. **SIMT**를 쓰는 CUDA에서는 스레드 하나의 관점에서 스칼라 덧셈을 작성한다. GPU는 한 워프의 참여 스레드에 같은 명령을 발행하고 각각의 피연산자로 계산한다.

| 구분 | SIMD | SIMT |
| --- | --- | --- |
| 의미 | Single Instruction, Multiple Data | Single Instruction, Multiple Threads |
| 프로그래밍 단위 | 벡터와 벡터 레인 | 개별 스레드 |
| 레지스터 상태 | 벡터 레지스터에 여러 값 | 스레드별 논리적 레지스터 상태 |
| 제어 흐름 | 벡터 마스크로 레인 선택 가능 | 스레드별 조건문과 하드웨어 실행 마스크 |
| 예시 | CPU의 AVX-512 | NVIDIA CUDA |

CPU와 GPU는 여러 방식의 병렬성을 함께 사용한다. CPU는 여러 스레드에서 SIMD 명령을 실행할 수 있고, GPU의 SIMT 모델도 하드웨어에서는 여러 레인이 실행 자원을 공유한다. [Intel · AVX-512 Instructions](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-instructions.html?page=1) · [CUDA Programming Guide · Programming Model](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/01-introduction/programming-model.html)

## 워프가 메모리를 기다리는 동안

워프 0이 메모리 읽기 명령을 발행했다. 다음 덧셈에는 읽어 올 값이 필요하므로 워프 0은 기다려야 한다. 그사이 스케줄러는 이미 SM에 올라와 있는 워프 1, 2, 3 중 다음 명령을 실행할 준비가 된 워프를 고른다. 워프 0이 값을 받는 시간은 그대로지만, 그 사이에 다른 워프의 계산이 진행된다. 이것이 **지연 시간 숨기기(latency hiding)**다.

![사이클 0에서 읽기를 요청한 워프 0이 사이클 9에서 덧셈을 발행할 때까지, 워프 1·2·3이 사이의 발행 슬롯을 채우는 타임라인.]({{ '/assets/images/blog/gpu-warps-occupancy/warp-scheduling.svg' | relative_url }})

그림은 스케줄러 하나가 준비된 워프의 명령을 고르는 예다. 실제 대기 시간과 발행 순서는 GPU 구조와 워프의 준비 상태에 따라 달라진다. [그림 2.13 · Arpan Pathak](https://arpanpathak.github.io/gpu-parallel-book/chapters/part1_foundations/ch02_gpu_hardware_first_principles.html#210-the-sm-in-action-time-slicing) · [Nsight Compute Profiling Guide · Scheduler Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)

## 점유율 100%인데도 쉬는 이유

**점유율(occupancy)**은 SM에 상주하는 활성 워프 수를 하드웨어가 허용하는 최대 상주 워프 수로 나눈 값이다.

\\[
\text{Occupancy} =
\frac{\text{Active Warps per SM}}{\text{Max Resident Warps per SM}}
\times 100\%
\\]

SM마다 최대 64개 워프가 상주할 수 있고 지금 32개가 올라와 있다면 점유율은 50%다. 여기서 활성 워프(active warp)는 SM에 상주하는 워프다. 메모리 응답을 기다리는 워프도 여기에 포함된다. **발행 가능 워프(eligible warp)**는 그중 지금 다음 명령을 발행할 준비가 된 워프다.

점유율이 100%여도 상주 워프가 모두 데이터를 기다리면 스케줄러는 새 명령을 발행할 수 없다. 점유율 50%에서도 준비된 명령이 꾸준히 있다면 높은 처리량을 낼 수 있다. 프로파일러에서 이론 점유율은 커널 설정과 자원 사용량으로 계산한 상한이고, 달성 점유율은 실행 중 관측한 값이다. 발행 가능한 워프가 부족한지 보려면 점유율과 함께 스케줄러의 발행 통계를 확인해야 한다. [CUDA Best Practices Guide · Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) · [Nsight Compute Profiling Guide · Scheduler Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)

| 상주 워프 수를 제한하는 요소 | 영향 |
| --- | --- |
| 레지스터 | 스레드당 사용량이 늘면 SM의 레지스터 파일에 들어갈 블록 수가 줄 수 있다. |
| 공유 메모리 | 블록당 사용량이 늘면 SM에 동시에 배치할 블록 수가 줄 수 있다. |
| 블록 크기 | 블록 하나가 차지하는 스레드 수와 워프 수에 따라 배치가 달라진다. |
| 하드웨어 상한 | GPU마다 SM당 최대 상주 블록·스레드·워프 수가 정해져 있다. |

점유율을 올리려고 레지스터 수를 제한하면 그 안에 담지 못한 값이 로컬 메모리로 스필되어 읽기·쓰기 명령이 늘 수 있다. 그래프에서 점유율이 올라가도 커널 시간이 길어질 수 있는 이유다. [CUDA Best Practices Guide · Register Pressure](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure)

## 실행할 일을 늘리는 두 가지 방법

**TLP(Thread-Level Parallelism)**는 여러 스레드의 작업을 활용하는 병렬성이다. 워프 0이 기다리는 동안 다른 워프의 명령을 발행하는 앞의 예가 여기에 해당한다.

**ILP(Instruction-Level Parallelism)**는 서로 의존하지 않는 명령을 활용한다. 같은 스레드가 두 값을 읽어 다음처럼 계산하는 경우다.

~~~cpp
float a = input[i];
float b = input[i + offset];

float x = a * 2.0f;
float y = b * 3.0f;

output[i] = x + y;
~~~

마지막 덧셈에는 x와 y가 모두 필요하다. 하지만 앞의 두 곱셈은 서로의 결과를 기다리지 않는다. a가 먼저 도착하면 b를 기다리는 중에도 a를 사용하는 계산을 진행할 여지가 있다. 실제 순서는 컴파일러가 만든 명령과 GPU의 발행 조건에 따라 달라진다.

스레드마다 더 많은 데이터를 맡기면 독립적인 명령이 늘 수 있다. 동시에 레지스터 사용량이 늘거나 메모리 접근 패턴이 달라질 수 있다. 읽고 쓰는 바이트 수는 데이터를 실제로 얼마나 옮기고 재사용하는지에 달려 있다. [CUDA Best Practices Guide · Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) · [Before GPU Engineering · Chapter 10](https://before-gpu-engineering.github.io/BEFORE-GPU-ENGINEERING/inside-the-silicon/chapter-10.html)

CUDA나 Triton 커널을 비교할 때는 먼저 **실행 시간과 유효 처리량**을 본다. 발행 슬롯이 자주 비면 준비된 워프가 부족한 이유를 찾고, 분기에서 실행 레인이 많이 빠지면 같은 워프의 데이터가 어떤 경로를 선택하는지 본다. 메모리가 병목이면 인접한 스레드의 접근이 적은 수의 메모리 트랜잭션으로 합쳐지는지(coalescing), 그리고 레지스터나 공유 메모리 사용량이 상주 워프 수를 얼마나 제한하는지 함께 살핀다. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) · [CUDA Best Practices Guide · Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

그림 세 장은 Arpan Pathak의 *GPU Architecture & Parallel Programming from First Principles*에서 가져왔다. [2장 원문](https://arpanpathak.github.io/gpu-parallel-book/chapters/part1_foundations/ch02_gpu_hardware_first_principles.html) · [MIT 라이선스]({{ '/assets/images/blog/gpu-warps-occupancy/LICENSE.txt' | relative_url }})
