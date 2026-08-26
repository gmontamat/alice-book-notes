## Vectorized gradient form of least squares linear regression

A linear model on $\mathbf{x} \sim (c)$ is defined as:

$$f(\mathbf{x}) = \mathbf{w}^{\top} \mathbf{x} + b$$

Where $\mathbf{w} \sim (c)$ and $b \in \mathbb{R}$.

The least-squares optimization problem is given by:

$$w^{\*}, b^{\*} = \underset{w, b}{\arg\min} \frac{1}{n} \sum_{i=1}^n
(y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2$$

We can vectorize the linear model by defining:

$$\mathbf{X} = \begin{bmatrix} \mathbf{x_1^{\top}} \\
\cdots \\
\mathbf{x_n^{\top}} \end{bmatrix} \sim (n, c)$$

Then, least-squares can be batched:

$$f(\mathbf{X}) = \mathbf{X w} + \mathbf{1} b \sim (n)$$

and the loss becomes:

$$LS(\mathbf{w}, b) = \frac{1}{n} \left\lVert \mathbf{y} - \mathbf
{X w} - \mathbf{1} b \right\rVert^2$$

$$LS(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{w}^{\top}
\mathbf{x}_i - b)^2 \in \mathbb{R}$$

Now, we can ignore the bias term $b$ if we assume the last component of each
$\mathbf{x}_i$ is $1$:

$$\mathbf{x} = \begin{bmatrix} \mathbf{x} \\
1 \end{bmatrix} \sim (c + 1)$$

Then, the loss becomes:

$$LS(\mathbf{w}) = \frac{1}{n} \left\lVert \mathbf{y} - \mathbf
{X w} \right\rVert^2$$

$$LS(\mathbf{w}) = \frac{1}{n} (\mathbf{y} - \mathbf{X w})^{\top}
(\mathbf{y} - \mathbf{X w})$$

$$LS(\mathbf{w}) = \frac{1}{n} (\mathbf{y}^{\top} \mathbf{y} - \mathbf{y}^{\top}
\mathbf{X w} - \mathbf{w}^{\top} \mathbf{X}^{\top} \mathbf{y} +
\mathbf{w}^{\top} \mathbf{X}^{\top} \mathbf{X w})$$

And its gradient:

$$\nabla LS(\mathbf{w}) = \frac{\partial LS(\mathbf{w})}{\partial \mathbf{w}} =
\frac{1}{n} (-2 \mathbf{X}^{\top} \mathbf{y} + 2 \mathbf{X}^{\top}
\mathbf{X w})$$

$$\nabla LS(\mathbf{w}) = \frac{2}{n} \mathbf{X}^{\top} (\mathbf{X w} -
\mathbf{y})$$

Ignoring the term $\frac{2}{n}$, we can simplify:

$$\nabla LS(\mathbf{w}) = \mathbf{X}^{\top} (\mathbf{X} \mathbf{w} - \mathbf
{y})$$

We can verify the dimensions: $\mathbf{X}^{\top} \sim (c, n)$, $\mathbf{X}
\mathbf{w} - \mathbf{y} \sim (n)$, so $\nabla LS(\mathbf{w}) \sim (c)$.

## Regularizing least-squares

We obtain the closed-form solution for $\mathbf{w}^{\*}$ (aka. no gradient
descent needed) by setting $\nabla LS(\mathbf{w}^{\*}) = \mathbf{0}$:

$$\mathbf{X}^{\top} (\mathbf{X} \mathbf{w}^{\*} - \mathbf{y}) = \mathbf{0}$$

$$\mathbf{X}^{\top} \mathbf{X} \mathbf{w}^{\*} - \mathbf{X}^{\top} \mathbf{y} =
\mathbf{0}$$

$$\mathbf{X}^{\top} \mathbf{X} \mathbf{w}^{\*} = \mathbf{X}^{\top} \mathbf{y}$$

$$(\mathbf{X}^{\top} \mathbf{X})^{-1} (\mathbf{X}^{\top} \mathbf{X})
\mathbf{w}^{\*} = (\mathbf{X}^{\top} \mathbf{X})^{-1} \mathbf{X}^{\top}
\mathbf{y}$$

$$\mathbf{w}^{\*} = (\mathbf{X}^{\top} \mathbf{X})^{-1} \mathbf{X}^{\top}
\mathbf{y}$$

If one feature (row in $\mathbf{X}$) is a scalar multiple of the other, then
$\mathbf{X}^{\top} \mathbf{X}$ is not invertible (**collinearity**). We can add
a small multiple ($\lambda > 0$) of $\mathbf{I} \sim (c, c)$ to invert the
matrix (note how this ensures there's no collinearity):

$$\mathbf{w}^{\*} = (\mathbf{X}^{\top} \mathbf{X} + \lambda \mathbf{I})^{-1}
\mathbf{X}^{\top} \mathbf{y}$$

This is known as the closed form solution for the **regularized least-squares**
or **ridge regression** loss:

$$LS_{\text{ridge}}(\mathbf{w}) = \left\lVert \mathbf{y} - \mathbf{X w}
\right\rVert^2 + \lambda \left\lVert \mathbf{w} \right\lVert^2$$

We can work out the closed-form solution above, let's expand the loss:

$$LS_{\text{ridge}}(\mathbf{w}) = (\mathbf{y} - \mathbf{X w})^{\top}
(\mathbf{y} - \mathbf{X w}) + \lambda \mathbf{w}^{\top} \mathbf{w}$$

$$LS_{\text{ridge}}(\mathbf{w}) = \mathbf{y}^{\top} \mathbf{y} -
\mathbf{y}^{\top} \mathbf{X w} - \mathbf{w}^{\top} \mathbf{X}^{\top}
\mathbf{y} + \mathbf{w}^{\top} \mathbf{X}^{\top} \mathbf{X w} +
\lambda \mathbf{w}^{\top} \mathbf{w}$$

We take the gradient and equal it to $\mathbf{0}$:

$$\nabla_{\mathbf{w}} LS_{\text{ridge}}(\mathbf{w}^{\*}) = -2 \mathbf{X}^{\top}
\mathbf{y} + 2 \mathbf{X}^{\top} \mathbf{X} \mathbf{w}^{\*} + 2 \lambda
\mathbf{w}^{\*} = 0$$

$$-\mathbf{X}^{\top} \mathbf{y} + \mathbf{X}^{\top} \mathbf{X} \mathbf{w}^{\*} +
\lambda \mathbf{w}^{\*} = 0$$

$$(\mathbf{X}^{\top} \mathbf{X} + \lambda \mathbf{I}) \mathbf{w}^{\*} =
\mathbf{X}^{\top} \mathbf{y}$$ 

$$\mathbf{w}^{\*} = (\mathbf{X}^{\top} \mathbf{X} + \lambda \mathbf{I})^{-1}
\mathbf{X}^{\top} \mathbf{y} \quad \blacksquare$$ 

The term $\lambda \left\lVert \mathbf{w} \right\lVert^2$ is known as $l_2$
regularization. It does not depend on the dataset and encodes a preference for
a certain type of solution (low-norm weights). The chapter does not mention the
Lasso regression that adds an $l_1$ penalty
term ($\lambda \left\lvert \mathbf{w} \right\rvert$) to the loss function and
does not have a closed form solution. It also favors sparse optimal weights,
driving weaker coefficients in $\mathbf{w}$ to exactly zero.
