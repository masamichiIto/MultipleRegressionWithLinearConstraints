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
In real-world analysis, it is often necessary to impose constraints on parameters. An example is when the sum of the regression coefficients of the explanatory variables is constrained to be 1 and interpreted as the relative presence of the contributions of the explanatory variables. (In this case, however, the objective can be achieved by simply estimating the regression coefficients as an unconstrained problem and then scaling them.)   

In this section, MR with such a constraint will be presented. Let's denote general linear constraint as $C\beta = t$, then the optimization probelem with this constraint is defined as follows;

$$
\begin{align*}
\underset{\beta}{argmin}\space &\|e\|^2 = \|y-X\beta\|^2 (=:f(\beta))\\
&subject\space to\space C\beta = t
\end{align*}
$$
```math
\begin{align*}
\underset{\beta}{argmin}\space &\|e\|^2 = \|y-X\beta\|^2 (=:f(\beta))\\
&subject\space to\space C\beta = t
\end{align*}
``` 
### 2-2. Estimation
Lagrange multiplier methods can be used in this setting in order to obtain estimator of the above constraind problem. Let $\lambda$ be a $p\times 1$ Lagrange multiplier vector, then objective function $g(\beta)$ to be minimized is defined as follows;
$$
g(\beta) := \|y-X\beta\|^2 + \lambda^T(C\beta - t)
$$
```math
g(\beta) := \|y-X\beta\|^2 + \lambda^T(C\beta - t)
```

Since $g(\beta)$ is also convex with respect to $\beta$, the solution of the equation with the partial derivative of $g(\beta)$ with respect to $\beta$ set to zero is the solution to this constrained optimization problem.

```math
\begin{align*}
\frac{\partial g(\beta)}{\partial \beta} &= -2X^Ty + 2X^TX\beta + C^T\lambda = 0\\
\Leftrightarrow \space \hat{\beta} &= (X^TX)^{-1}X^Ty -\frac{1}{2}(X^TX)^{-1}C^T\lambda\\
&= \hat{\beta}_{LS}-\frac{1}{2}(X^TX)^{-1}C^T\lambda
\end{align*}
```

Here, since $\hat{\beta}$ should satisfy the constraint $C\beta = t$, we obtain the following formula for $\lambda$ by premultiplying both sides of the equation of $\hat{\beta}$ by $C$.

$$
t = C\hat{\beta} = C\hat{\beta}_{LS}-\frac{1}{2}C(X^TX)^{-1}C^T\lambda
$$

```math
\begin{align*}
t &= C\hat{\beta} = C\hat{\beta}_{LS}-\frac{1}{2}C(X^TX)^{-1}C^T\lambda\\
\Leftrightarrow & \lambda = 2\left\{C(X^TX)^{-1}C^T\right\}^{-1}(C\hat{\beta} - t)\\
\end{align*}
```

From the above, the estimator of the coefficient vector $\beta$ with linear constraint $C\beta = t$ is as follows.

$$
\hat{\beta}_{lc} = \hat{\beta}_{LS}-(X^TX)^{-1}C^T\left\{C(X^TX)^{-1}C^T\right\}^{-1}(C\hat{\beta} - t)
$$

```math
\hat{\beta}_{lc} = \hat{\beta}_{LS}-(X^TX)^{-1}C^T\left\{C(X^TX)^{-1}C^T\right\}^{-1}(C\hat{\beta} - t)
```

## 3. MR with Stochastic Constraints
### 3-1. Model
### 3-2. Estimation

## Reference
- https://www.kwansei.ac.jp/s_sociology/kiyou/38/38_ch05.pdf
    - <--(in Japanese)