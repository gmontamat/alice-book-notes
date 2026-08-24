## Vectorized gradient form of least squares linear regression

A linear model on $\mathbf{x} \sim (c)$ is defined as:

$$f(\mathbf{x}) = \mathbf{w}^{\top} \mathbf{x} + b$$

Where $\mathbf{w} \sim (c)$ and $b \in \mathbb{R}$.

The least-squares optimization problem is given by:

$$w^{\*}, b^{\*} = \underset{w, b}{\arg\min} \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2$$

We can vectorize the linear model by defining:

$$\mathbf{X} = \begin{bmatrix} \mathbf{x_1^{\top}} \\ ... \\ \mathbf{x_n^{\top}} \end{bmatrix} \sim (n, c)$$
