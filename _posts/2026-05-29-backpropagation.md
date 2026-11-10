---
title: "Backpropagation 정리: Chain Rule로 가중치를 업데이트하는 방법"
date: 2026-05-29 13:40:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - study
tags:
  - DeepLearning
  - Backpropagation
  - Gradient
source: "Notion PDF Export - Back propagation"
---


# Backpropagation: Loss가 각 Weight에 미친 영향 구하기

Backpropagation은 loss가 각 weight와 bias에 얼마나 영향을 받았는지 계산하는 방법이다. Output에서 시작해 입력 방향으로 gradient를 전달하고, 그 값을 이용해 parameter를 업데이트한다.

문제는 모든 parameter를 같은 크기로 바꿀 수 없다는 점이다. 어떤 weight는 loss에 큰 영향을 주고, 어떤 weight는 거의 영향을 주지 않는다.

핵심은 chain rule이다. Forward pass에서 여러 함수를 차례로 통과했다면, backward pass에서는 각 함수의 미분을 반대 순서로 곱해 입력 쪽 parameter의 gradient를 구한다.

## Chain Rule

Chain Rule은 미분에서 연쇄법칙이라고도 불린다. 먼저 수학적으로 정의해보자.

함수 $f$, $g$가 있을 때 $f$와 $g$가 모두 미분 가능하고 $F=f(g(x))=f \circ g$로 정의된 합성 함수이면 $F$도 미분 가능하다.

이때 $F'(x)=f'(g(x)) \cdot g'(x)$이다.

$t=g(x)$라고 한다면, $\frac{dy}{dx} = \frac{dy}{dt} \frac{dt}{dx}$이 성립한다.

---

수식만 보면 흐름이 잘 보이지 않는다.

합성함수는 한 함수의 출력이 다른 함수의 입력으로 들어가는 구조다. 아래 코드로 보면 단순하다.

```python
def f(g):
    return g * 3

def g(x):
    return x + 1

x = 3
F = f(g(x))
# F = 12
```

미분은 입력을 조금 바꿨을 때 출력이 얼마나 바뀌는지 구하는 과정이다. 위 코드에서 $x$가 바뀌면 $g(x)$가 바뀌고, 이어서 최종 $F$도 바뀐다. Chain rule은 이 변화가 여러 함수를 지나며 어떻게 이어지는지 계산한다.

```python
F = f(g(4))
# F = 15
F = f(g(2))
# F = 9
```

즉 chain rule은 $X$가 변했을 때 함수 $g$가 얼마나 변하는지, 그리고 $g$의 변화로 함수 $f$가 얼마나 변하는지 연결해서 보는 방법이다. 이 연쇄적인 변화를 알 수 있으면 값 $F$의 변화에 함수 $f$와 $g$가 얼마나 기여했는지 계산할 수 있다.

그렇다면 함수 여러 개가 들어가면 어떻게 될까?

---

이변수 함수 $z=f(x,y)$에서 $x=h(s,t), y=g(s,t)$일 때 

$f(x,y), g(s,t), h(s,t)$가 모두 미분 가능하면

$$
\frac{\partial z}{\partial s}
= \frac{\partial z}{\partial x}\frac{\partial x}{\partial s}
+ \frac{\partial z}{\partial y}\frac{\partial y}{\partial s}
,\quad
\frac{\partial z}{\partial t}
= \frac{\partial z}{\partial x}\frac{\partial x}{\partial t}
+ \frac{\partial z}{\partial y}\frac{\partial y}{\partial t}
$$

나타낼 수 있다.

$s$나 $t$가 얼마나 변했는지에 따라 함수 $z$의 변화량을 위 식처럼 구할 수 있다.

그리고 $\partial$는 편미분을 뜻하는 기호다. 기준이 되는 변수 하나만 남기고 나머지 변수는 상수처럼 두고 미분한다.

---

## Forward propagation

Backpropagation이 이뤄지기 전에는 forward propagation을 진행해야 한다. 초기화한 $w$ 값과 input인 $x$를 가지고 순전파 계산을 진행한 후, 우리가 원하는 값이 나오는지, 나오지 않았다면 얼마나 차이 나는지 구해야 한다.

![backprop forward]({{ '/assets/images/blog/backprop-forward.png' | relative_url }})

이 모델은 2개의 input, 2개의 output, 2개의 hidden layer를 가진 neural network이다.

![backprop backward]({{ '/assets/images/blog/backprop-backward.png' | relative_url }})

먼저 target output으로 $y_1$에 0.2, $y_2$에 0.7을 넣고, input으로는 $x_1$에 0.2, $x_2$에 0.5를 넣는다. 각 $w$는 임의의 값으로 초기화했다.

이 계산에서는 activation function으로 sigmoid, loss function으로 MSE를 사용한다.

먼저 layer 0에서 받을 값부터 계산해보자. 보통 행렬로 계산한다.

$$
\begin{aligned}z_{10} &= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \times \begin{bmatrix} w_{100} \\ w_{200} \end{bmatrix} \\z_{11} &= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \times \begin{bmatrix} w_{110} \\ w_{210} \end{bmatrix}\end{aligned}
$$

저 행렬 곱을 풀어보면 아래 식과 같고 결국 $wx$들의 합의 형태이다.

$$
\begin{aligned}z_{10} &= x_1 w_{100} + x_2 w_{200} \\&= (0.2 \times 0.1) + (0.5 \times 0.3) \\&= 0.02 + 0.15 = 0.17 \\\\z_{11} &= x_1 w_{110} + x_2 w_{210} \\&= (0.2 \times 0.2) + (0.5 \times 0.1) \\&= 0.04 + 0.05 = 0.09\end{aligned}
$$

$z_{10}$와 $z_{11}$의 값을 구했으면 activation function에 넣어 $a_{10}$과 $a_{11}$ 값을 구해보자.

사용할 sigmoid의 수식과 코드는 아래와 같다.

$$
\sigma = \frac{1}{1 + e^{-x}}
$$

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

$$
\begin{aligned}a_{10} &= \sigma(z_{10}) = 0.54 \\a_{11} &= \sigma(z_{11}) = 0.52\end{aligned}
$$

다음 레이어도 위와 같이 구해보자.

$$
\begin{aligned}&\text{[Layer 0]} && \text{[Layer 1]} \\z_{10} &= 0.17 \quad \to \quad a_{10} = 0.54 \quad & z_{20} &= 0.27 \quad \to \quad a_{20} = 0.57 \\z_{11} &= 0.09 \quad \to \quad a_{11} = 0.52 \quad & z_{21} &= 0.43 \quad \to \quad a_{21} = 0.61\end{aligned}
$$

$y_{1}$과 $y_{2}$는 각각 $a_{20}$과 $a_{21}$로 output을 구했다. 우리가 원하는 $y_{1}$과 $y_{2}$는 0.2와 0.7이었는데, 계산된 output은 0.57과 0.61이라 차이가 있다.

그럼 MSE 함수로 loss 값을 구해야 한다. 얻기 바라는 값을 $t$, 실제로 나온 값을 $y$라고 할 때 loss는 다음과 같다.

$$
E = \frac{1}{2} \sum (t_i - y_i)^2
$$

결과와 원하는 값이 얼마나 차이 나는지 구하고 그 값들의 평균을 내는 것이다. 학습시킨다는 것은 loss 값을 0에 가깝게 만드는 것이다. 여기서 나온 loss 값을 backpropagation하면 된다.

```python
def MSE(targets, values):
    if not isinstance(values, list):
        return False

    result = 0
    for target, value in zip(targets, values):
        result += 0.5 * (target - value) ** 2

    return result

print(MSE([0.2, 0.7], [0.57, 0.61]))  # 결과: 0.0725
```

이제 구한 loss 값으로 역전파를 진행해보자.

## Back Propagation

![backprop backward]({{ '/assets/images/blog/image.png' | relative_url }})


0.4로 할당되어 있는 가중치 $w_{10}^{1}$의 값을 업데이트할 것이다. 그러려면 $w_{10}^{1}$이 전체 에러 $E$에 얼마나 영향을 미쳤는지, 즉 기여도를 구해야 한다. 이때 chain rule이 사용된다.

$E$에 대해 $w_{10}^{1}$이 얼마나 영향을 미쳤는지 기여도 식으로 풀면 아래와 같다.

$$
\frac{\partial E}{\partial w_{10}^{1}} = \frac{\partial E}{\partial a_{20}} \frac{\partial a_{20}}{\partial z_{20}} \frac{\partial z_{20}}{\partial w_{10}^{1}}
$$

먼저 $\frac{\partial E}{\partial a_{20}}$부터 풀어보면, 원래 구한 $E$는 아래와 같은 식이다.

$$
E = \frac{1}{2} ((t_1 - a_{20})^2 + (t_2 - a_{21})^2)
$$

여기서 $a_{20} = y_1, a_{21} = y_2$이기 때문에 치환한다. 하지만 $\frac{\partial E}{\partial a_{20}}$는 편미분 식이므로, 지금 구하려는 값과 상관없는 $a_{21}$은 0으로 생각하고 풀면 된다.

$$
\frac{\partial E}{\partial a_{20}} = (t_{1} - a_{20}) \times -1 + 0 = (0.2 - 0.57) \times -1 = 0.37
$$

이 계산을 통해 $E$에 대해 $a_{20}$, 즉 $y_1$이 0.37만큼 기여한 것을 알 수 있다. 이런 식으로 계속해보자.

$$
\frac{\partial a_{20}}{\partial z_{20}} = a_{20} \times (1 - a_{20}) = 0.57 \times (1 - 0.57) = 0.25
$$

$$
\frac{\partial z_{20}}{\partial w_{10}^{1}} = a_{10} + 0 = 0.54
$$

$$
\frac{\partial E}{\partial w_{10}^{1}} = 0.37 \times 0.25 \times 0.54 = 0.049
$$

최종적으로 $E$에 $w_{10}^{1}$가 기여한 값은 0.049로 계산된다. 이 값을 학습식에 넣으면 $w_{10}^{1}$ 값을 업데이트할 수 있다.

이때 값을 얼마나 크게 움직일지, 얼마나 빨리 학습할지를 정하는 값이 learning rate다. 여기서는 0.3으로 잡는다.

$$
w_{10}^{1+} = w_{10}^{1} - (L \times \frac{\partial E}{\partial w_{10}^{1}}) = 0.4 - (0.3 \times 0.049) = 0.3853
$$

이렇게 해서 새로운 $w_{10}^{1}$ 값을 구했다. 다른 weight와 bias도 같은 방식으로 gradient를 계산해 업데이트한다.

## 내가 이해한 핵심

Backpropagation은 loss를 줄이는 방향을 직접 찾아주는 별도의 규칙이 아니다. Forward pass에서 사용한 계산을 반대로 따라가며, 각 parameter가 loss에 기여한 정도를 chain rule로 구하는 과정이다.

```text
Forward
input -> weighted sum -> activation -> output -> loss

Backward
loss -> output gradient -> activation gradient -> weight gradient
```

핵심은 다음 update 식이다.

$$
w_{new} = w_{old} - \eta \frac{\partial E}{\partial w}
$$

Gradient는 어느 방향으로 움직여야 loss가 커지는지를 나타낸다. 그래서 반대 방향으로 learning rate만큼 이동하면 loss를 줄일 수 있다.
