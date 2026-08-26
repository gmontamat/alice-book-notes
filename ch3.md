## Statistical learning theory

Concerned with the behavior of:

$$f^{\*} = \underset{f}{\arg\min} \frac{1}{n} \sum_{i=1}^{n} l(y_i,
f(\mathbf{x}_i))$$
<p align="right">(1)</p>

When seen as a finite-sample approximation of:

$$\mathbb{E}_{p(\mathbf{x}, y)}[l(y, f(\mathbf{x}))]$$
<p align="right">(2)</p>

In ML, we don't minimize across the space of all possible functions, our models
are parametrized by $\mathbf{w}$. We then train $f(\mathbf{x}, \mathbf{w})$ via
gradient descent:

$$\mathbf{w}^{\*} = \underset{\mathbf{w}}{\arg\min} \frac{1}{n} \sum_{i=1}^{n}
l(y_i, f(\mathbf{x}_i, \mathbf{w}))$$
<p align="right">(3)</p>

The difference between the expected ($2$) and empirical ($1$) loss is called
the **generalization gap**. A memorization algorithm will minimize *1* but have
poor generalization (**overfit** the specific training data). Counter-intuitive
properties of modern neural networks (strong generalization long after
overfitting should have been expected) have opened many avenues of research in
SLT.

## Loss Minimization vs Maximum Likelihood

**Loss Minimization**: Directly optimizes a predefined loss
  function $L(\mathbf{w})$ that quantifies prediction errors. Approach is
  task-specific and can include arbitrary penalties (e.g. $l_1$/$l_2$
  regularization) without probabilistic interpretation.

**Maximum Likelihood**: Maximizes the joint probability $P(\mathscr{S}_n|f)$ of
  observing the training data under a model distribution. Provides
  probabilistic foundation and naturally yields negative log-likelihood:
  $\log P(\mathscr{S}_n|f) = -\sum_i \log P(y_i|\mathbf{x}_i;\mathbf{w})$.

**Key Relationship**: When the loss function is the negative log-likelihood,
  both approaches are mathematically equivalent. ML offers principled
  uncertainty quantification and asymptotic properties
  (consistency, efficiency), while arbitrary loss functions offer more
  flexibility for specific objectives.

**Bottom Line**: ML is loss minimization with a probabilistically meaningful
  loss function.
