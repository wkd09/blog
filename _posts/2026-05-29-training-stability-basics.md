---
title: "딥러닝 학습 안정화 기초: Activation, Bias-Variance, Overfitting"
date: 2026-05-29 00:50:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - study
tags:
  - AI
  - Deep Learning
  - Optimization
source: "Notion PDF Export - Activation Function, Variance & Bias, Overfitting"
---

딥러닝 모델을 학습할 때는 모델 구조만 중요한 것이 아니다. 어떤 activation function을 쓰는지, 모델이 너무 단순하거나 복잡하지 않은지, 학습 데이터와 테스트 데이터 사이에서 일반화가 되는지도 중요하다.

이 글에서는 activation function, bias-variance trade-off, overfitting과 underfitting을 함께 정리한다.

## Activation Function이 필요한 이유

뉴런은 기본적으로 다음과 같은 선형 계산을 한다.

$$
z = wx + b
$$

만약 신경망이 선형 함수만 사용한다면, layer를 아무리 많이 쌓아도 결국 하나의 선형 함수와 같아진다. 복잡한 패턴을 학습하려면 비선형성이 필요하다.

Activation function은 이 비선형성을 넣어주는 역할을 한다.

![activation functions]({{ '/assets/images/blog/activation-functions.png' | relative_url }})

## Sigmoid
![sigmoid]({{ '/assets/images/blog/sigmoid.png' | relative_url }})

Sigmoid는 출력을 0과 1 사이로 만든다.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

확률처럼 해석할 수 있어 이진 분류의 출력층에서 사용된다. 하지만 입력값이 너무 크거나 작으면 gradient가 거의 0이 되어 vanishing gradient 문제가 생길 수 있다.


## Tanh

Tanh는 출력을 -1과 1 사이로 만든다.
![tanh]({{ '/assets/images/blog/tanh.png' | relative_url }})

Sigmoid보다 출력 중심이 0에 가까워 학습이 조금 더 잘 될 수 있다. 하지만 여전히 vanishing gradient 문제가 있다.

## ReLU
![relu]({{ '/assets/images/blog/relu.png' | relative_url }})

ReLU는 딥러닝에서 가장 널리 쓰이는 activation function이다.

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

계산이 단순하고 양수 구간에서는 gradient가 유지되기 때문에 sigmoid나 tanh보다 깊은 네트워크 학습에 유리하다.

단점은 입력이 음수인 경우 출력이 0이 되고 gradient도 흐르지 않는 dead neuron 문제가 생길 수 있다는 점이다.

## Leaky ReLU

![leaky relu]({{ '/assets/images/blog/leaky-relu.png' | relative_url }})

Leaky ReLU는 ReLU의 dead neuron 문제를 완화하기 위해 음수 구간에도 작은 기울기를 남긴다.

음수 입력을 완전히 0으로 만들지 않기 때문에 일부 neuron이 영구적으로 죽는 문제를 줄일 수 있다.

## Softmax

Softmax는 여러 class에 대한 확률 분포를 만든다.

출력값은 0과 1 사이이고, 전체 합은 1이다. 그래서 다중 클래스 분류의 출력층에서 자주 사용된다.

## Bias와 Variance

Bias는 예측값의 평균과 실제 값 사이의 차이를 의미한다. Bias가 크면 모델이 데이터를 너무 단순하게 보고 있다는 뜻이다.

Variance는 예측값이 얼마나 흩어져 있는지를 의미한다. Variance가 크면 학습 데이터에는 잘 맞지만, 새로운 데이터에서는 성능이 흔들릴 수 있다.

## Bias-Variance Trade-off

모델이 너무 단순하면 bias가 커지고 underfitting이 발생한다. 반대로 모델이 너무 복잡하면 variance가 커지고 overfitting이 발생한다.

![bias variance curves]({{ '/assets/images/blog/bias-variance-curves.png' | relative_url }})
![bias variance target]({{ '/assets/images/blog/bias-variance-target.png' | relative_url }})

정리하면 다음과 같다.

- Low Bias, Low Variance: 가장 이상적인 상태
- High Bias, Low Variance: underfitting 가능성이 큼
- Low Bias, High Variance: overfitting 가능성이 큼
- High Bias, High Variance: 학습도 일반화도 잘 안 되는 상태

좋은 모델은 학습 데이터를 충분히 설명하면서도 테스트 데이터에 일반화될 수 있어야 한다.

## Overfitting

Overfitting은 학습 데이터에 과하게 맞춰진 상태다.

학습 데이터에서는 성능이 좋지만, 테스트 데이터나 실제 서비스 데이터에서는 성능이 떨어진다. 학습 데이터가 부족하거나 모델 capacity가 너무 큰 경우 자주 발생한다.

해결 방법은 다음과 같다.

- 모델 capacity 낮추기
- dropout 사용
- L1/L2 regularization 적용
- 학습 데이터 늘리기
- data augmentation 적용

## Underfitting

Underfitting은 학습 데이터조차 제대로 학습하지 못한 상태다.

원인은 다음과 같다.

- epoch이 너무 적음
- 모델이 너무 단순함
- feature 표현력이 부족함
- 데이터가 부족하거나 전처리가 부적절함

Underfitting은 모델 capacity를 늘리거나 학습을 더 진행하거나 feature를 개선하는 방식으로 완화할 수 있다.

## 내가 이해한 핵심

딥러닝 학습 안정화는 여러 요소가 함께 작동한다.

- Activation function은 비선형성을 넣어 복잡한 패턴을 학습하게 한다.
- ReLU는 깊은 네트워크에서 기본 선택지로 자주 사용된다.
- Bias가 크면 underfitting, variance가 크면 overfitting을 의심할 수 있다.
- 모델 capacity와 regularization을 조절해 일반화 성능을 맞춰야 한다.

모델을 개선할 때는 loss만 보지 말고, 학습/검증 성능 차이와 activation, capacity, regularization을 함께 봐야 한다.
