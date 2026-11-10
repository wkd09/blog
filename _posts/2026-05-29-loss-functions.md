---
title: "Loss Function 정리: 모델이 얼마나 틀렸는지 측정하는 방법"
date: 2026-05-29 13:20:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - study
tags:
  - ML
  - Loss
  - Optimization
source: "Notion PDF Export - Loss Function"
---

# Loss Function: 모델이 얼마나 틀렸는지 하나의 값으로 만들기

모델을 학습하려면 먼저 예측이 얼마나 틀렸는지 숫자로 표현해야 한다. 이 값을 만드는 함수가 loss function이다.

```text
input -> model -> prediction
                    ↓
target ----------> loss
```

Loss가 작으면 예측이 target에 가깝고, 크면 멀다는 뜻이다. Optimizer는 이 loss가 줄어드는 방향으로 weight와 bias를 업데이트한다.

문제는 task마다 “틀렸다”의 의미가 다르다는 점이다. 집값 예측에서는 실제값과의 거리 자체가 중요하지만, 이미지 분류에서는 정답 class에 얼마나 높은 확률을 줬는지가 중요하다.

이 글에서는 회귀에서 사용하는 MSE, RMSE, MAE와 분류에서 사용하는 cross-entropy를 정리한다.

## Loss Function의 역할

Loss function은 크게 두 가지 역할을 한다.

1. **학습 방향을 만든다.** Backpropagation은 loss를 각 parameter로 미분해 gradient를 구한다.
2. **학습 상태를 확인한다.** Train loss와 validation loss를 비교하면 학습 진행, overfitting과 underfitting을 확인할 수 있다.

중요한 점은 loss와 실제 서비스 metric이 항상 같지는 않다는 것이다. 예를 들어 classification은 cross-entropy로 학습하면서 accuracy나 F1 score로 평가할 수 있다. Loss는 gradient를 만들기 좋아야 하고, metric은 최종 목표를 잘 보여줘야 한다.

## 문제 유형에 따라 무엇을 사용할까

| 문제 | 대표 Loss | 비교하는 것 |
| --- | --- | --- |
| 이진 분류 | Binary Cross-Entropy | 정답 0/1과 class 1의 예측 확률 |
| 다중 분류 | Categorical Cross-Entropy | 정답 분포와 전체 class 확률 분포 |
| 회귀 | MSE, MAE | 실제 연속값과 예측값의 거리 |

같은 task라도 outlier의 영향, label 표현, 원하는 평가 기준에 따라 loss를 다르게 선택할 수 있다.

## Cross-Entropy Loss

Cross-Entropy는 분류에서 사용하는 대표적인 loss다. 모델이 정답 class에 높은 확률을 주면 loss가 작아지고, 확신을 갖고 틀린 class를 선택하면 loss가 크게 증가한다.

![cross entropy loss]({{ '/assets/images/blog/cross-entropy-loss.png' | relative_url }})

### Binary Cross-Entropy

Binary Cross-Entropy(BCE)는 정답이 0 또는 1인 이진 분류에 사용한다. 모델 출력 $p_i$는 sample $i$가 class 1일 확률로 본다.

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left(y_i\log p_i + (1-y_i)\log(1-p_i)\right)
$$

![binary cross entropy math]({{ '/assets/images/blog/binary-cross-entropy-math.png' | relative_url }})

- $N$: sample 수
- $y_i$: 실제 label, 0 또는 1
- $p_i$: class 1이라고 예측한 확률

정답이 1이면 $-\log p_i$가 남고, 정답이 0이면 $-\log(1-p_i)$가 남는다. 정답 확률이 1에 가까우면 loss는 0에 가까워진다.

### Categorical Cross-Entropy

Categorical Cross-Entropy는 class가 여러 개인 분류에 사용한다. Output dimension은 class 수와 같고, softmax를 거친 각 값은 해당 class일 확률로 해석한다.

$$
\mathcal{L}_{\text{CCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij}\log p_{ij}
$$

![categorical cross entropy math]({{ '/assets/images/blog/categorical-cross-entropy-math.png' | relative_url }})

- $N$: sample 수
- $C$: 클래스 수
- $y_{ij}$: sample $i$의 one-hot label
- $p_{ij}$: sample $i$가 class $j$라고 예측한 확률

One-hot label에서 정답 위치만 1이므로 실제로는 정답 class의 $-\log p$만 loss에 남는다.

### Sparse Categorical Cross-Entropy

Sparse Categorical Cross-Entropy는 같은 계산을 사용하지만 label을 one-hot vector가 아닌 정수 class index로 받는다.

$$
\mathcal{L}_{\text{SCCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
$$

![sparse categorical cross entropy math]({{ '/assets/images/blog/sparse-cross-entropy-math.png' | relative_url }})

- $y_i$: sample $i$의 정수 class index
- $p_{i,y_i}$: 정답 class에 할당한 확률

두 categorical loss의 차이는 label 표현 방식이다. 계산하려는 목표 자체는 같다.

## 회귀에서 사용하는 Loss

회귀는 class가 아니라 연속된 값을 예측한다. 따라서 실제값과 예측값 사이의 거리를 직접 계산한다.

![mse loss]({{ '/assets/images/blog/mse-loss.png' | relative_url }})

### MSE(Mean Squared Error)

MSE는 오차를 제곱한 뒤 평균을 구한다.

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

![mse math]({{ '/assets/images/blog/mse-math.png' | relative_url }})

- $N$: sample 수
- $y_i$: 실제값
- $\hat{y}_i$: 모델이 예측한 값

제곱을 사용하므로 오차의 부호가 사라지고 큰 오차에 더 강한 penalty를 준다. Outlier에 민감하지만 큰 실수를 적극적으로 줄이고 싶을 때 유용하다.

### RMSE(Root Mean Squared Error)

RMSE는 MSE에서 제곱근을 취한 형태로, 오차를 실제값과 동일한 단위로 표현하는 데 유용하다.

$$
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
$$

![rmse math]({{ '/assets/images/blog/rmse-math.png' | relative_url }})

MSE는 단위도 제곱된다. RMSE는 다시 제곱근을 취하므로 원래 target과 같은 단위로 오차를 볼 수 있다. 다만 학습 loss보다 사람이 결과를 해석하는 evaluation metric으로 자주 사용한다.

### MAE(Mean Absolute Error)

MAE는 예측값과 실제값 사이의 차이를 절댓값으로 변환한 후 평균을 구하는 방식이다.

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

![mae math]({{ '/assets/images/blog/mae-math.png' | relative_url }})

- $|y_i - \hat{y}_i|$: sample의 절대 오차

MAE는 제곱하지 않으므로 MSE보다 outlier의 영향을 덜 받는다. 모든 오차가 크기에 비례해 loss에 반영된다.

| 항목 | MSE | MAE |
| --- | --- | --- |
| 큰 오차 | 제곱으로 강하게 반영 | 크기에 비례해 반영 |
| Outlier 영향 | 큼 | 상대적으로 작음 |
| Gradient | 0 근처에서 부드러움 | 0에서 미분 처리 필요 |

## 내가 이해한 핵심

Loss function은 단순한 성적표가 아니다. Model parameter를 어느 방향으로 움직일지 결정하는 학습 신호다.

```text
분류
정답 class에 얼마나 높은 확률을 줬는가
-> Cross-Entropy

회귀
실제값과 예측값이 얼마나 떨어졌는가
-> MSE / MAE
```

핵심은 관습적으로 loss를 고르는 것이 아니라 task에서 어떤 실수를 더 크게 벌줄지 정하는 것이다. 큰 오차를 강하게 줄이고 싶으면 MSE가 맞을 수 있고, outlier에 덜 흔들리고 싶으면 MAE가 더 적절할 수 있다.
