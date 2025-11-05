- Attention은 NMT(Neural Machine Translation: 신경망 기계 번역) 분야에서 처음 등장한 이론이다.
- 아래 구조를 Attention Mechanism이라고 한다

- Attention은 일종의 방법론이고 이를 구현하는 다양한 방법이 존재한다
	- 15개 이상의 Attention구현 논문이 있음

# Attention의 등장 배경
- 이전까지는 RNN을 이용하여 자연어를 처리함
- 하지만 RNN을 사용하니 입력 문장이 길어질 수록 인코더의 마지막 출력인 컨텍스트 벡터가 너무 많은 값을 압축하여 정확한 정보를 가지지 못하고 초기의 값이 기울기 손실로인해 출력에서 영향이 거의 없어지고 마지막 단어가 많은 영향을 받는 문제가 발생함
	- seq2seq 방법의 문제점
- 즉, 입력문장이 길어지면 문장의 전체 컨텍스트를 반영하지 못하는 문제가 발생함
> 그래서 디코더에서 문장을 생성할 때 입력 문장의 전체 컨텍스트가 온전하게 전달 될 수 있도록 하는 방법을 고안함

# 최초의 Attention: Bahdanau Attention
- Dzmitry Bahdanau가 쓴 'Neural Machine Translation by Jointly Learning to Align and Translate' 논문에 등장
	- 여기 공동 저자가 조경현 교수님,  요슈아 벤지오(Yoshua Bengio)
## 핵심 원리

$$
p(y_i|y_1, \dots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, \mathbf{s}_i, c_i),
$$

- 이전에 생성된 단어와 전체 문장 X에 대하여 $y_i$번쨰 단어가 나올 조건부 확률
	- 기존의 인코더 디코더 모델은 전체를 보지 않고 인코더의 출력값과 이전 단어의 히든 스테이트만 보았음
	
- $y_i$가 나올 확률을 출력하는 비선형 함수
	- $y_{i-1}$는 이전 단어
	- $s_i$은 $i$번쨰 시점에서 의 RNN의 히든 상태
	- $c_i$는 i번째 단어를 생성하기 위해서 사용되는 컨텍스트 정보

$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$

$$
현재 단어 = f(이전\ 상태, 이전\ 단어, 현재\ 컨텍스트\ 정보)
$$

## $C_i$ 계산하기
$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$
 - $T_x$는 입력 문장의 전체 길이
 - $j$는 입력 문장에서 위치
 - $\alpha_{ij}$는 i번쨰 값에 대한 $j$번째 $h$에 대한 가중치
 - $h_j$입력 문장에서 j번째 위치한 값의 히든 스테이트
 > 모든 입력 위치에서의 히든 스테이트의 가중합을 컨텍스트 정보로 사용하겠다
> => 모든 입력 단어에 대해 현재 i번째 출력 단어와의 관계를 계산하여 컨텍스트 정보로 사용하겠다.

### $h_i$계산하기
$$
h_t = f(x_t, h_{t-1})
$$

$$
\vec{h}'_i = \begin{cases}
    (1 - \vec{z}_i) \circ \vec{h}'_{i-1} + \vec{z}_i \circ \underline{\vec{h}'_i} & \text{if } i > 0 \\
    0 & \text{if } i = 0
\end{cases}
$$

$$
\begin{align*}
\vec{\underline{h}}_i &= \tanh \left( \vec{W} Ex_i + \vec{U} \left[ \vec{r}_i \circ \vec{h}_{i-1} \right] \right) \\
\vec{z}_i &= \sigma \left( \vec{W}_z Ex_i + \vec{U}_z \vec{h}_{i-1} \right) \\
\vec{r}_i &= \sigma \left( \vec{W}_r Ex_i + \vec{U}_r \vec{h}_{i-1} \right).
\end{align*}
$$

- 복잡한데 사실 그냥 LSTM레이어 지나간다는 수식인
	- LSTM은 그냥 RNN의 기울기 소실 문제를 해결하기 위해서 장기 메모리와 단기 메모리를 만들어서 이용하겠다는 신경망 구조임

$$
h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]
$$

- 이해하기 쉽게 

- 왼쪽부터 현재 위치 j까지 히든스테이트를 LSTM을 거쳐서 계산한 값 과
- 오른쪽부터 현재 위치 j까지 히든 스테이트를 LSTM을 거쳐서 계산된 값을 연결한 값
	- 그냥 두 벡터은 하나의 백터로 만듬(뒤에 붙인다)

### $a_{ij}$ 계산하기

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})},
$$

=> 해석하면 1부터 $T_x$까지  i에 대한 각 j번쨰 단어의 $e_{ij}$에 대한 소프트맥스 값
- i는 출력 문장에서 단어 위치 , j는 입력 문장에서 단어 위치
- 즉, 현재 예측하려는 출력 문장에서 i번째 단어에 대해 모든 입력 문장의 단어의 $e_{ij}$의 값들에 대하여 소프트맥스를 취하겠다.
### $e_{ij}$ 계산하기 aka. score
- i번째 문자에 대한 j번째 문자와 관계 점수

$$
e_{ij} = a(s_{i-1}, h_j)
$$

- 논문에서 위치 j 주변의 입력과 위치 i에서의 출력이 얼마나 잘 일치하는지 점수
- a를 이를 계산하는 alignment 모델이라고 함

$$
\begin{gather*} 
e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j) \\
v_{a} \in \mathbb{R}^{n'}\\
W_{a} \in \mathbb{R}^{n' \times n}\\
U_{a} \in \mathbb{R}^{n' \times 2n}\\
\end{gather*}
$$

$$
(n'*n) \times (n*1) + (n' * 2n) \times (2n*1) = n' * 1
$$

$$
{(n'*1)}^T \times n'*1 = (1 * n') \times n' * 1 = R
$$
- 이때 계산되는 실수 값이 입력 문장의 j번쨰 단어와 출력단어 i의 관계를 나타내는 점수

## 다시 $C_i$계산하기
- $e_{ij}$는 스칼라 값 => 소프트맥스 => 0 과 1 사이의 실수 값
- j가 1부터 $T_x$의 까지 
	- $a_{ik}h_{k}$ ->스칼라 값 곱하기 2n차원 백터 => 2n차원
	- $a_{ij}h_{j}$의 합 -> 2n차원
	- 사실 그냥 가중합임
- 2n차원의 2n 크기의 컨텍스트 벡터
## $s_i$계산하기
- s는 디코더 부분에 있는 LSTM 구조라서 LSTM에 입력으로 $(s_{i-1}, y_{i - 1}, c_i)$를 넣으면 $s_i$값이 나온다.
# 다음 단어 예측하기
- $g(y_{i-1}, s_i , c_i)$의 결과가 다음 단어를 나타내는 벡터임
	- g는 다층 신경망임