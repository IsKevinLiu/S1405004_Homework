# 统计学习原理课程作业

## 一、Lasso问题简介

&emsp;&emsp;在有监督的模型学习中，我们往往希望我们的模型预测值与实际值$b$尽可能地接近，即考虑优化问题

$$
\min \frac{1}{2}\parallel Ax-b \parallel_2^2
$$

其中, $x$是模型参数, $Ax$是模型预测值, $b$是实际观测值. 但在一些问题处理中, 通过上述优化问题所得到的模型可能存在过拟合的现象. 同时, 在面对高维数据时, 该方法也往往表现不佳. 为了克服上述问题, 可以在优化问题的目标函数中引入正则化项作为惩戒项, 进而控制模型的复杂度, 即

$$
\min \frac{1}{2}\parallel Ax-b \parallel_2^2 + P(x)
$$

而在Lasso问题中, 则通过引入模型参数的1-范数充当正则化项, 即

$$
\min \frac{1}{2}\parallel Ax-b \parallel_2^2 + \lambda\parallel x\parallel_1
$$

由于存在非光滑项 $\parallel x\parallel_1$ 使得目标函数不可微, 所以在求解过程中需要使用一些特殊的优化算法来进行求解. 例如: 近端梯度下降法, 交替方向乘子法等等.

## 二、Lasso问题的求解方法

### 0. 预备知识

#### 0.1 李普希兹常数 Lipschitz constant

&emsp;&emsp;给定一个函数 $f:\R^n \to \R$，如果存在一个常数 $L>0$，对于所有 $x,y \in \R^n$，都有：

$$
\parallel \nabla f(x)-\nabla f(y) \parallel <= L \parallel x-y \parallel
$$

那么我们称 $f$ 的梯度是李普希兹连续的, 且 $L$ 是 $f$ 的梯度的李普希兹常数.

#### 0.2 软阈值算子

&emsp;&emsp;设有一个实数 $\lambda>0$ (阈值参数), 对于任意实数 $x$, 软阈值算子 $S_\lambda(x)$ 定义为:

$$
S_\lambda(x) = 
\begin{cases} 
x - \lambda, & \text{if } x > \lambda \\
 0, & \text{if } |x| \leq \lambda \\
 x + \lambda, & \text{if } x < -\lambda 
\end{cases}
$$

#### 0.3 KKT条件

对于Lasso问题, 可以表示为

$$
\min_{x,y} \frac{1}{2}\parallel y \parallel^2_2+\lambda\parallel x\parallel_1 \\
\text{s.t. } Ax-b-y=0
$$

其Lagrange函数为

$$
L(x,y;\mu) = \frac{1}{2}\parallel y \parallel^2_2+\lambda\parallel x\parallel_1 - <\mu,Ax-b-y>
$$

由KKT条件可知, 其最优解满足

$$
\begin{aligned}
& \nabla_x L(x,y;\mu) = \lambda \partial \parallel x \parallel_1 - A^\text{T} \mu \ni 0 \\
& \nabla_y L(x,y;\mu) = y+\mu = 0 \\
& Ax-b-y = 0
\end{aligned}
$$

### 1. 近端梯度下降法

### 2. 加速近端梯度下降法

