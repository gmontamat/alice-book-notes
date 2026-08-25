## Vectorized gradient form of least squares linear regression

A linear model on $\mathbf{x} \sim (c)$ is defined as:

$$f(\mathbf{x}) = \mathbf{w}^{\top} \mathbf{x} + b$$

Where $\mathbf{w} \sim (c)$ and $b \in \mathbb{R}$.

The least-squares optimization problem is given by:

$$w^{\*}, b^{\*} = \underset{w, b}{\arg\min} \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2$$

We can vectorize the linear model by defining:

$$\mathbf{X} = \begin{bmatrix} \mathbf{x_1^{\top}} \\
\cdots \\
\mathbf{x_n^{\top}} \end{bmatrix} \sim (n, c)$$

Then, least-squares can be batched:

$$f(\mathbf{X}) = \mathbf{X w} + \mathbf{1} b \sim (n)$$

and the loss becomes:

$$LS(\mathbf{w}, b) = \frac{1}{n} \left\lVert \mathbf{y} - \mathbf{X w} - \mathbf{1} b \right\rVert^2$$

$$LS(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2 \in \mathbb{R}$$

Now, we can ignore the bias term $b$ if we assume the last component of each $\mathbf{x}_i$ is $1$:

$$\mathbf{x} = \begin{bmatrix} \mathbf{x} \\
1 \end{bmatrix} \sim (c + 1)$$

Then, the loss becomes:

$$LS(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{w}^{\top} \mathbf{x}_i)^2$$

And its gradient:

$$\nabla LS(\mathbf{w}) = \frac{\partial LS(\mathbf{w})}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^n 2 (y_i - \mathbf{w}^{\top} \mathbf{x}_i) (-\mathbf{x}_i)$$

$$\nabla LS(\mathbf{w}) = \frac{2}{n} \sum_{i=1}^n \mathbf{x}_i (\mathbf{w}^{\top} \mathbf{x}_i - y_i)$$

Note that $\mathbf{w}^{\top} \mathbf{x}_i - y_i \in \mathbb{R}$. So ignoring the term $\frac{2}{n}$, we can write in matrix form:

$$\nabla LS(\mathbf{w}) = \mathbf{X}^{\top} (\mathbf{X} \mathbf{w} - \mathbf{y})$$
