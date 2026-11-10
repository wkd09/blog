---
title: "Gradient Descent와 Optimizer 정리"
date: 2026-05-29 13:30:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - study
tags:
  - ML
  - GradientDescent
  - Optimizer
source: "Notion PDF Export - Gradient Descent, Optimizer"
---

![Gradient descent curve]({{ '/assets/images/blog/gradient-descent.svg' | relative_url }})

<small>Image: [Wikimedia Commons - Gradient descent](https://commons.wikimedia.org/wiki/Special:FilePath/Gradient_descent.svg)</small>

Gradient Descent는 loss를 줄이기 위해 파라미터를 조금씩 움직이는 최적화 방법이다. 현재 위치에서 loss가 가장 빠르게 증가하는 방향이 gradient라면, 학습은 그 반대 방향으로 이동한다.

$$
\theta_{t+1}
= \theta_t - \eta \nabla_{\theta}J(\theta_t)
$$

여기서 $\theta$는 파라미터, $\eta$는 learning rate, $J(\theta)$는 loss function이다.

## Learning Rate

Learning rate는 한 번 업데이트할 때 얼마나 크게 이동할지 정한다. 너무 작으면 학습이 매우 느리고, 너무 크면 최소점을 지나쳐 발산할 수 있다.

![learning rate size]({{ '/assets/images/blog/learning-rate-size.png' | relative_url }})

좋은 learning rate는 빠르게 내려가면서도 안정적으로 수렴하는 값이다. 실제 학습에서는 scheduler를 사용해 초반에는 크게, 후반에는 작게 조절하기도 한다.

학습 로그를 볼 때 loss가 거의 줄지 않으면 learning rate가 너무 작거나 gradient가 잘 전달되지 않는 상황일 수 있다. 반대로 loss가 갑자기 크게 튀거나 `NaN`이 나오면 learning rate가 너무 큰 경우가 많다.

실무에서는 처음부터 하나의 값을 고정하기보다 learning rate range test로 적절한 구간을 찾고, cosine decay나 step decay 같은 scheduler를 붙여 학습 후반을 안정화한다.

## Local Minimum과 Global Minimum

최적화의 목표는 loss가 가장 낮은 global minimum을 찾는 것이다. 하지만 복잡한 모델에서는 loss landscape가 울퉁불퉁하기 때문에 local minimum이나 saddle point 근처에서 학습이 느려질 수 있다.

![gradient descent surface]({{ '/assets/images/blog/gradient-descent-surface.png' | relative_url }})
![local minima]({{ '/assets/images/blog/local-minima.png' | relative_url }})

딥러닝에서는 파라미터 공간이 매우 크기 때문에 단순히 하나의 local minimum만 문제가 되는 것은 아니다. 평평하고 일반화가 잘 되는 영역으로 가는 것이 더 중요할 때가 많다.

## SGD

Stochastic Gradient Descent는 전체 데이터가 아니라 일부 mini-batch로 gradient를 계산한다.

![stochastic gradient descent]({{ '/assets/images/blog/stochastic-gradient-descent.png' | relative_url }})
![gradient descent batch]({{ '/assets/images/blog/gradient-descent-batch.png' | relative_url }})

장점은 계산이 빠르고, gradient noise가 있어 local minimum을 빠져나오는 데 도움이 될 수 있다는 점이다. 단점은 업데이트가 불안정할 수 있다는 점이다.

Mini-batch 크기도 optimizer의 동작에 영향을 준다. batch size가 작으면 업데이트가 noisy해져 일반화에 도움이 될 수 있지만, 너무 작으면 학습이 불안정하다. batch size가 크면 GPU를 효율적으로 쓰지만, sharp minimum으로 수렴해 일반화가 나빠질 수 있다는 논의도 있다.

## Momentum

Momentum은 이전 업데이트 방향을 일부 유지한다.

![momentum optimizer]({{ '/assets/images/blog/momentum-optimizer.png' | relative_url }})

$$
v_t = \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta_t)
$$

관성처럼 움직이기 때문에 같은 방향으로 계속 내려가는 경우 속도가 붙고, gradient가 자주 흔들리는 방향은 완화된다.

## AdaGrad

AdaGrad는 자주 등장하는 feature에는 작은 learning rate를, 드물게 등장하는 feature에는 큰 learning rate를 적용한다.

희소한 데이터에는 유리하지만, 누적 gradient가 계속 커져 learning rate가 너무 작아질 수 있다. 학습 후반에 거의 움직이지 않는 문제가 생길 수 있다.

## RMSProp

RMSProp은 AdaGrad의 learning rate 감소 문제를 완화하기 위해 gradient 제곱의 이동 평균을 사용한다. 오래된 gradient의 영향은 줄이고 최근 gradient를 더 반영한다.

이 방식은 non-stationary한 loss landscape에서 더 안정적으로 동작한다.

## Adam

Adam은 Momentum과 RMSProp을 결합한 optimizer다. gradient의 1차 moment와 2차 moment를 모두 추적한다.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

Adam은 기본값만으로도 잘 동작하는 경우가 많아서 딥러닝에서 널리 쓰인다. 다만 항상 최선은 아니며, task에 따라 SGD with momentum이나 AdamW가 더 좋은 일반화 성능을 보일 수 있다.

AdamW는 Adam에서 weight decay를 더 올바르게 분리한 optimizer다. Transformer 계열 모델을 학습하거나 fine-tuning할 때 AdamW가 기본 선택지처럼 쓰이는 이유가 여기에 있다.

## Scheduler와 Warmup

Optimizer만큼 중요한 것이 learning rate schedule이다. 특히 Transformer처럼 큰 모델은 학습 초반에 gradient가 불안정할 수 있어 warmup을 자주 사용한다.

Warmup은 처음 몇 step 동안 learning rate를 작은 값에서 시작해 점점 키우는 방식이다. 이후에는 cosine decay나 linear decay로 learning rate를 줄인다. 이 방식은 초반 발산을 막고 후반 수렴을 안정적으로 만든다.

## Gradient Clipping

RNN이나 큰 Transformer를 학습할 때 gradient가 갑자기 커지는 exploding gradient 문제가 생길 수 있다. Gradient clipping은 gradient norm이 특정 값보다 커지면 그 크기를 제한한다.

이 방법은 optimizer의 방향성을 완전히 바꾸기보다, 한 번의 업데이트가 지나치게 커지는 것을 막는 안전장치에 가깝다. loss가 갑자기 튀는 학습에서 자주 확인해야 하는 옵션이다.

## Optimizer 선택 기준

처음 실험할 때는 AdamW를 기본값으로 두는 것이 무난하다. 빠르게 수렴하고 hyperparameter에 비교적 덜 민감하기 때문이다.

하지만 이미지 분류처럼 SGD with momentum이 강한 domain도 있다. 최종 성능이 중요하면 AdamW로 빠르게 baseline을 만들고, 이후 SGD 계열이나 scheduler 조합을 비교하는 방식이 현실적이다.

## 내가 이해한 핵심

Gradient Descent는 loss를 줄이는 방향을 찾는 기본 원리이고, optimizer는 그 이동 방식을 더 안정적이고 빠르게 만드는 방법이다. learning rate, momentum, adaptive scaling을 어떻게 조합하느냐가 학습 안정성과 최종 성능에 큰 영향을 준다.
