# Statistical learning theory

Concerned with the behavior of:

$$f^{\*} = \underset{f}{\arg\min} \frac{1}{x} \sum_{i=1}^{n} l(y_i, f(x_i)) {eq1}$$

When seen as a finite-sample approximation of:

$$\mathbb{E}_{p(x, y)}[l(y, f(x))] {eq2}$$

In ML, we don't minimize across the space of all possible functions, our models
are parametrized by $w$. We then train $f(x, w)$ via gradient descent:

$$w^{\*} = \underset{w}{\arg\min} \frac{1}{x} \sum_{i=1}^{n} l(y_i, f(x_i, w)) {eq3}$$

The difference between the expected (*eq2*) and empirical (*eq1*) loss is called
the **generalization gap**. A memorization algorithm will minimize *eq1* but have poor
generalization (**overfit** the specific training data).
