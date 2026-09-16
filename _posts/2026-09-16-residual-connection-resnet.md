---
title: "Residual Connection 정리: 깊은 신경망은 왜 ResNet에서 더 잘 학습될까"
date: 2026-09-16 01:10:00 +0900
categories:
  - study
tags:
  - AI
  - DeepLearning
  - CNN
  - ResNet
  - Transformer
  - Optimization
source: "Deep Residual Learning for Image Recognition (He et al., 2016)"
---

신경망은 보통 깊어질수록 더 복잡한 패턴을 학습할 수 있다. 그런데 레이어를 계속 추가하면 어느 순간부터 성능이 좋아지기는커녕 학습 자체가 어려워진다. Residual Connection(잔차 연결)은 이 문제를 해결해 ResNet처럼 100층이 넘는 신경망도 안정적으로 학습할 수 있게 만든 핵심 구조다.

핵심은 간단하다. 블록이 입력 전체를 새로 변환하도록 두는 대신, 입력은 지름길로 그대로 전달하고 **입력에서 얼마나 바꿀지**만 학습하게 한다.

$$
y = F(x) + x
$$

여기서 $x$는 블록의 입력, $F(x)$는 convolution, normalization, activation 등을 거친 변환 결과다. 모델은 전체 함수 $H(x)$가 아니라 $H(x)-x$인 잔차(residual) $F(x)$를 학습한다.

## 깊은 모델에서 생기는 degradation problem

직관적으로는 56층 모델이 20층 모델보다 최소한 같거나 더 좋은 성능을 내야 한다. 20층 모델을 그대로 두고, 추가한 레이어가 항등함수(identity mapping)를 학습하면 되기 때문이다.

하지만 일반적인 plain network는 항등함수조차 잘 학습하지 못할 수 있다. 특히 convolution과 ReLU 같은 비선형 연산을 여러 번 통과하면서, 추가한 레이어를 “아무 변화도 만들지 않는 층”으로 만드는 최적화가 쉽지 않다.

![20층과 56층 plain network의 학습 및 테스트 오류](/assets/images/blog/resnet-degradation-20-vs-56.png)

위 그래프에서 56층 plain network는 20층 모델보다 training error와 test error가 모두 높다. 이는 test 성능만 나빠지는 overfitting과 다르다. **훈련 데이터에도 더 잘 맞추지 못하는 최적화 문제**가 degradation problem이다.

Vanishing gradient와도 완전히 같은 문제는 아니다. 기울기 소실은 역전파 중 gradient가 0에 가까워져 앞쪽 레이어가 학습되지 않는 현상이고, degradation은 깊이를 늘렸을 때 더 좋은 해를 찾지 못하는 더 넓은 최적화 난점이다. Residual Connection은 두 문제를 모두 완화하는 데 도움을 준다.

## Residual block의 구조

Residual block은 입력을 두 갈래로 보낸다.

1. **변환 경로**: 입력 $x$가 여러 weight layer를 통과해 $F(x)$를 만든다.
2. **skip 경로**: 입력 $x$를 변형하지 않고 블록 끝까지 전달한다.
3. **합산**: 두 경로의 값을 원소별로 더해 $F(x)+x$를 만든다.

![Residual block: 변환 경로 F(x)와 identity skip 경로 x](/assets/images/blog/resnet-residual-block.png)

입력과 출력의 차원이 같다면 skip 경로에는 학습 파라미터가 필요 없다. 채널 수나 spatial size가 달라 직접 더할 수 없을 때는 $W_s$로 입력을 projection한다.

$$
y = F(x, W) + W_s x
$$

즉, 기존 정보를 유지하는 기본 경로는 이미 주어져 있다. 블록은 “무엇을 새로 만들 것인가”보다 “기존 표현에서 무엇을 보정할 것인가”에 집중한다.

## 왜 학습이 쉬워질까

### 1. 항등함수를 구조적으로 제공한다

어떤 블록이 현재 표현을 굳이 바꿀 필요가 없다면 $F(x)\approx0$만 학습하면 된다. Plain network가 복잡한 weight와 nonlinearity를 조합해 $H(x)=x$를 만들어야 하는 것보다 쉽다.

이 덕분에 깊은 모델은 얕은 모델의 해를 포함할 수 있다. 필요한 블록만 의미 있는 변환을 학습하고, 불필요한 블록은 입력을 거의 그대로 통과시키면 된다.

### 2. gradient가 지름길로도 흐른다

출력 $y=F(x)+x$를 입력에 대해 미분하면 다음과 같다.

$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1
$$

역전파 때 `+1`에 해당하는 identity 경로가 존재하므로, 변환 경로의 gradient가 작아지더라도 정보와 gradient가 더 직접적으로 전달될 수 있다. 이것이 매우 깊은 네트워크의 학습 안정성에 기여한다.

### 3. 여러 길이의 경로를 만든다

블록마다 변환 경로와 skip 경로 중 하나를 지날 수 있으므로, ResNet은 다양한 길이의 계산 경로를 갖는다. 이를 여러 개의 상대적으로 얕은 네트워크가 모인 ensemble처럼 해석하기도 한다.

모든 블록을 반드시 지나는 긴 경로만 쓰는 것이 아니라, 학습에 유효한 짧은 경로를 통해 gradient가 전달될 수 있다는 관점이다. 다만 이는 Residual Connection의 유용성을 설명하는 해석 중 하나이며, 효과를 완전히 하나의 이론으로 설명할 수 있는 것은 아니다.

## Plain network와 ResNet의 차이

![plain-18과 plain-34, ResNet-18과 ResNet-34의 오류 비교](/assets/images/blog/resnet-plain-vs-resnet-errors.png)

왼쪽의 plain network에서는 34층이 18층보다 높은 오류를 보인다. 깊이를 추가했지만 최적화가 더 어려워진 것이다.

오른쪽의 ResNet에서는 같은 비교에서 34층 모델의 오류가 18층보다 낮다. Residual Connection이 깊이를 실제 성능 향상으로 연결한 결과다.

## ImageNet에서의 결과

![ImageNet 분류 오류 비교 표](/assets/images/blog/resnet-imagenet-results.png)

ResNet은 ImageNet 대규모 이미지 분류에서 당시 대표 모델이던 VGG와 GoogLeNet 계열을 앞섰다. 표의 ResNet-152는 top-1 error **19.38%**, top-5 error **4.49%**를 기록했다.

이 결과의 더 중요한 의미는 단순히 점수가 좋아졌다는 데 있지 않다. 기존 모델보다 훨씬 깊은 152층 네트워크를 실용적으로 학습할 수 있음을 보여줬다는 점이다.

## Transformer에서도 쓰이는 이유

Residual Connection은 CNN의 ResNet에서 널리 알려졌지만, Transformer에서도 각 블록의 기본 구성 요소다. Self-Attention과 Feed-Forward Network의 출력에 각각 원래 입력을 더한다.

$$
z = x + \mathrm{Attention}(x)
$$

$$
y = z + \mathrm{FFN}(z)
$$

이후 Layer Normalization을 함께 사용해 activation 규모를 안정화한다. Transformer 역시 매 블록이 완전히 새로운 표현을 만들기보다, 기존 토큰 표현을 보존하면서 필요한 변화만 누적하는 구조다.

## 내가 이해한 핵심

Residual Connection은 단순히 레이어를 건너뛰는 연결이 아니다. 깊은 네트워크가 어려운 전체 변환을 한 번에 배우지 않고, 이미 있는 표현에 대한 **변화량만 점진적으로 학습하도록 문제를 바꾸는 구조**다.

- 입력을 직접 전달해 항등함수를 쉽게 구현할 수 있다.
- gradient가 더 짧고 안정적인 경로로 전달될 수 있다.
- 깊이를 늘렸을 때 발생하는 degradation problem을 완화한다.
- ResNet뿐 아니라 Transformer, U-Net 등 현대 딥러닝 아키텍처의 기본 구성 요소가 되었다.

## 참고

- Kaiming He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016.
- [Residual connection은 왜 효과적일까?](https://channelai.tistory.com/2)
- [Attention Is All You Need 3 - Residual Connection](https://cheorish.tistory.com/33)
