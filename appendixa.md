## Maximum likelihood estimator of the Bernoulli distribution

Consider the case of a Bernoulli distribution with unknown parameter $p$.

Then, $P(X=x) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$.

Given $n$ samples $x_i, i \in \{1 ,..., n\}$ drawn from this distribution,
the maximum likelihood estimator is:

$$p^* = \underset{p}{\arg\max} \sum_{i=1}^{n} \log(p^x_i (1-p)^{1-x_i})$$

$$p^* = \underset{p}{\arg\max} \sum_{i=1}^{n} x_i \log(p) + (1 - x_i) \log(1-p)$$

Using the fact that this function is convex in $p$, we can find the maximum, $p^*$,
in terms of $x_i$ by taking the derivative and equaling to $0$:

$$\frac{\partial}{\partial p} \sum_{i=1}^{n} x_i \log(p) + (1 - x_i) \log(1-p) = 0$$

$$\sum_{i=1}^{n} \frac{x_i}{p^{\*}} - \frac{1-x_i}{1 - p^{\*}} = 0$$

$$\sum_{i=1}^{n} \frac{x_i (1 - p^{\*}) - p^{\*} (1 - x_i)}{p^{*} (1 - p^{\*})} = 0$$

$$\sum_{i=1}^{n} \frac{x_i - p^{\*}}{p^{\*} (1 - p^{\*})} = 0$$

Assuming $p^{\*} \ne 0$:

$$\sum_{i=1}^{n} x_i - p^{\*} = 0$$

$$\sum_{i=1}^{n} x_i = \sum_{i=1}^{n} p^{\*} = n p^{\*}$$

Thus,

$$p^{\*} = \frac{\sum_{i=1}^{n} x_i}{n}$$
