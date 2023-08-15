# MultipleRegressionWithLinearConstraints
Python script for the implementation for multiple regression with linear constraints will be developed.

## 1. Preliminaries - Multiple Regression -
### 1-1. Model
Multiple Regression(MR) is used to express linear relationships between dependent variable and independent variables. Let $y$ be a $n\times 1$ dependent vector(target), $X$ be a $n\times p$ matrix(independent variables), $\beta$ be a $p\times 1$ coefficient vector, then MR model is expressed as follows.

```math
\begin{align*}
&y = X\beta + e\\
&where \space e\sim N(0_p, \sigma^2I_p)
\end{align*}
```
### 1-2. Estimation
#### a. Least Squares Method
In the case of the least-squares method, the coefficient vector is estimated by minimizing the sum of squares of the errors. The optimization problem in this case is as follows.

```math
\begin{align*}
\underset{\beta}{argmin}\space &\|e\|^2 = \|y-X\beta\|^2 (=:f(\beta))\\
&where\space \|x\| := x^T x = \sum_{i}x_i^2
\end{align*}
```
The objective function $f(\beta)$ is obviously convex with respect to $\beta$, and by setting the partial derivative of the function with respect to $\beta$ to $0$, we obtain the following equation (known as the normal equation).

```math
\begin{align*}
\frac{\partial f(\beta)}{\partial \beta} &= 0\\
\Leftrightarrow X^TX\beta &= X^Ty
\end{align*}
```

Assuming that $X^TX$ is full rank, the least-squares estimator is as follows.
```math
\hat{\beta}_{LS} = (X^TX)^{-1}X^Ty
```

#### b. Maximum Likelihood Method
Setting $e := y-Xbeta$, since $E[e]=0_p, V[e]=\sigma^2 I_p$ from the assumption, the log likelihood $l(\beta)$ is defined as follows.

```math
\begin{align*}
l(\beta) &= \log \prod_{i=1}^nf(e_i|\beta) = \sum_{i=1}^n\log f(e_i|\beta)\\
&= \log \left(\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2}(y-X\beta)^T(\sigma^2 I_p)^{-1}(y-X\beta)\right)\right)\\
&= \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta)\right)\right)\\
&= \left(-\frac{1}{2}\log2\pi\sigma^2 -\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta)\right)\\
&\propto -\|y-X\beta\|^2 = -f(\beta)
\end{align*}
```
Since ML estimator is obtained as maximizer of $l(\beta)$, ML estimator is equivalent to LS estimator in this setting, i.e. $\beta_{ML} = \beta_{LS} = (X^TX)^{-1}X^Ty$ .  

## 2. MR with Linear Constraints
### 2-1. Model
### 2-2. Estimation


## 3. MR with stochastic Constraints
### 3-1. Model
### 3-2. Estimation

## Reference
- https://www.kwansei.ac.jp/s_sociology/kiyou/38/38_ch05.pdf
    - <--(in Japanese)