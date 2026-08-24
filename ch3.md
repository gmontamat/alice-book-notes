# Statistical learning theory

Concerned with the behavior of:

$$f^{\*} = \underset{f}{\arg\min} \frac{1}{x} \sum_{i=1}^{n} l(y_i, f(x_i))$$ (1)

When seen as a finite-sample approximation of:

$$\mathbb{E}_{p(x, y)}[l(y, f(x))]$$ (2)

In ML, we don't minimize across the space of all possible functions, our models
are parametrized by $w$. We then train $f(x, w)$ via gradient descent:

$$w^{\*} = \underset{w}{\arg\min} \frac{1}{x} \sum_{i=1}^{n} l(y_i, f(x_i, w))$$ (3)

The difference between the expected (*2*) and empirical (*1*) loss is called
the **generalization gap**. A memorization algorithm will minimize *1* but have poor
generalization (**overfit** the specific training data). Counter-intuitive properties
of modern neural networks (strong generalization long after overfitting should have
been expected) have opened many avenues of research in SLT.

## Loss Minimization vs Maximum Likelihood

**Loss Minimization**: Directly optimizes a predefined loss function $L(w)$ that
quantifies prediction errors. Approach is task-specific and can include arbitrary
penalties (e.g. L1/L2 regularization) without probabilistic interpretation.

**Maximum Likelihood**: Maximizes the joint probability $P(\mathscr{S}|f)$ of
observing the training data under a model distribution. Provides probabilistic
foundation and naturally yields negative
log-likelihood: $\log P(\mathscr{S}|f) = -\sum_i \log P(y_i|x_i;w)$.

**Key Relationship**: When the loss function is the negative log-likelihood, both
approaches are mathematically equivalent. ML offers principled uncertainty
quantification and asymptotic properties (consistency, efficiency), while arbitrary
loss functions offer more flexibility for specific objectives.

**Bottom Line**: ML is loss minimization with a probabilistically meaningful loss
function.
