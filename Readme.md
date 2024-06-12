# 统计学习原理课程作业

## 一、Lasso问题简介

&emsp;&emsp;在有监督的模型学习中，我们往往希望我们的模型预测值与实际值$b$尽可能地接近，即考虑优化问题

$$
\min \frac{1}{2}\parallel Ax-b \parallel_2^2
$$

其中, $x$ 是模型参数, $Ax$ 是模型预测值, $b$ 是实际观测值. 但在一些问题处理中, 通过上述优化问题所得到的模型可能存在过拟合的现象. 同时, 在面对高维数据时, 上述方法也会面临维数灾难的问题. 为了克服上述问题, 可以在优化问题的目标函数中引入正则化项作为惩戒项, 进而控制模型的复杂度, 即

$$
\min \frac{1}{2}\parallel Ax-b \parallel_2^2 + P(x)
$$

而在Lasso问题中, 则通过引入模型参数的1-范数充当正则化项, 即

$$
\min \frac{1}{2}\parallel Ax-b \parallel_2^2 + \lambda\parallel x\parallel_1
$$

由于存在非光滑项 $\parallel x\parallel_1$ 使得目标函数不可微, 所以在求解过程中需要使用一些特殊的优化算法来进行求解. 例如: 近端梯度下降法, 交替方向乘子法等等.

## 二、Lasso问题的求解方法

### 2.0 预备知识

#### 2.0.1 李普希兹常数(Lipschitz Constant)

&emsp;&emsp;给定一个函数 $f:\mathbb{R}^n \to \mathbb{R}$，如果存在一个常数 $L>0$，对于所有 $x,y \in \mathbb{R}^n$，都有：

$$
\parallel \nabla f(x)-\nabla f(y) \parallel \le L \parallel x-y \parallel
$$

那么我们称 $f$ 的梯度是李普希兹连续的, 且 $L$ 是 $f$ 的梯度的李普希兹常数. 当函数 $f(x)$ 的梯度满足L-Lipschitz条件时, 则其在定点 $x^{(k)}$ 处的二阶近似值为:

$$
\begin{aligned}
f(x) &\approx f(x^{(k)}) + <\nabla{f(x^{(k)}),x-x^{(k)}}> + \frac{1}{2}<\nabla^2f(x^{(k)}), \parallel x-x^{(k)}\parallel^2>\\
&\le f(x^{(k)}) + <\nabla{f(x^{(k)}),x-x^{(k)}}> + \frac{L}{2}\parallel x-x^{(k)} \parallel^2\\
&= \frac{L}{2} \parallel x-(x^{(k)}-\frac{1}{L}\nabla f(x^{(k)}))\parallel^2 + C(x^{(k)}) = \hat{f}(x)
\end{aligned}
$$

其中, $C(x^{(k)})$ 是与 $x^{(k)}$ 有关的常数

#### 2.0.2 软阈值算子(Soft Thresholding Operator)

&emsp;&emsp;设有一个实数 $\lambda>0$ (阈值参数), 对于任意实数 $x$, 软阈值算子 $S_\lambda(x)$ 定义为:

$$
S_\lambda(x) = 
\begin{cases} 
x - \lambda, & \text{if } x > \lambda \\
 0, & \text{if } |x| \leq \lambda \\
 x + \lambda, & \text{if } x < -\lambda 
\end{cases}
$$

#### 2.0.3 KKT条件

&emsp;&emsp;对于Lasso问题, 可以表示为

$$
\min_{x,y} \frac{1}{2}\parallel y \parallel^2_2+\lambda\parallel x\parallel_1 \\
\text{s.t. } Ax-b-y=0
$$

其Lagrange函数为

$$
L(x, y;\mu) = \frac{1}{2}\parallel y \parallel^2_2+\lambda\parallel x\parallel_1 - <\mu, Ax-b-y>
$$

由KKT条件可知, 其最优解满足

$$
\begin{aligned}
& \nabla_x L(x, y;\mu) = \lambda \partial \parallel x \parallel_1 - A^\text{T} \mu \ni 0 \\
& \nabla_y L(x, y;\mu) = y+\mu = 0 \\
& Ax-b-y = 0
\end{aligned}
$$

对广义方程求解,

$$
\begin{aligned}
& A^\text{T}\mu \in \lambda \partial \parallel x \parallel_1 \\
& A^\text{T}\mu +x \in (I + \lambda \partial \parallel \cdot \parallel_1)(x) \\
& x \in (I + \lambda \partial \parallel \cdot \parallel_1)^{-1}(A^\text{T}\mu +x) \\
&x=\text{Prox}_{\lambda\parallel \cdot \parallel_1} (A^\text{T}\mu +x)
\end{aligned}
$$

即

$$
\text{Prox}_{\lambda\parallel \cdot \parallel_1} (A^\text{T}\mu +x) - x = 0
$$

则利用KKT条件可以设置停机标准:

$$
\max\{ \frac{\parallel \text{Prox}_{\lambda\parallel \cdot \parallel_1} (A^\text{T}\mu +x) - x\parallel}{1+\parallel x \parallel} , \frac{\parallel y+\mu \parallel}{1+\parallel y \parallel} , \frac{\parallel Ax-b-y \parallel}{1+\parallel b \parallel} \} \le \text{tol}
$$

其中, $\text{tol}$是收敛容限.

### 2.1 近端梯度下降法

#### 2.1.1 推导

当目标函数形式为 $\min_x f(x)$ , 其中 $f(x)$ 是可微凸函数时, 利用梯度下降法的迭代公式为:

$$
x^{(k+1)} = x^{(k)}-\alpha_k\nabla f(x^{(k)})
$$

其中, $\alpha_k$ 是第 $k$ 次迭代时的步长. 并且当函数 $f\in\text{LC}^1$ 时, 上式可以写为 $x^{(k+1)} = x^{(k)}-\frac{1}{L}\nabla f(x^{(k)})$. 但对于目标函数为

$$
\min_x f(x)=g(x)+h(x)
$$

其中, $f,g$ 是凸函数, 且 $f \in \text{LC}^1$. $h(x)$ 是非光滑的, 则需要利用近端算子来处理非光滑部分. 非光滑算子定义为:

$$
\text{Prox}_{}
$$

### 2. 加速近端梯度下降法


### 3. ADMM
