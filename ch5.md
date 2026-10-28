## Composition of linear models is a linear model

Suppose we chaned together two different linear models:

$$\mathbf{h} = \textit{f}_1(\mathbf{x}) =
\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$

$$y = \textit{f}_2(\mathbf{h}) =
\mathbf{w}_2^{\top} \mathbf{h} + b_2$$

Where $x \sim (c_1)$, $\mathbf{W}_1 \sim (c_2, c_1)$, $\mathbf{b}_1 \sim (c_2)$,
$\mathbf{h} \sim (c_2)$, $\mathbf{w}_2 \sim (c_2)$, and $b_2, y \in \mathbb{R}$.

If we compose the functions:

$$\textit{f}_2(\textit{f}_1(\mathbf{x})) = \mathbf{w}_2^{\top}
(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + b_2$$

$$\textit{f}_2(\textit{f}_1(\mathbf{x})) =
\mathbf{w}_2^{\top} \mathbf{W}_1 \mathbf{x} + \mathbf{w}_2^{\top} \mathbf{b}_1 +
b_2$$

If we define $\mathbf{A} = \mathbf{w}_2^{\top} \mathbf{W}_1 \sim (c_1)^{\top}$
and $c = \mathbf{w}_2^{\top} \mathbf{b}_1 + b_2 \in \mathbb{R}$, then:

$$\textit{f}_2(\textit{f}_1(\mathbf{x})) = \mathbf{A} \mathbf{x} + c$$

Therefore, composing linear models yields the same results as using a single
plain linear model.

## Universal approximation of MLPs

Given a continuous function $g : \mathbb{R}^d \rightarrow \mathbb{R}$, there
exists an MLP with a single layer model $f(\mathbf{x})$ with an element-wise
sigmoid activation function ($\phi$) of the form:

$$\mathbf{h} = f_1(\mathbf{x}) = \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$

$$y = f_2(\mathbf{h}) = \mathbf{w}_2^{\top} \mathbf{h} + b_2$$

Such that for any $\epsilon > 0$:

$$\lvert f(\mathbf{x}) - g(\mathbf{x})) \rvert \le \epsilon \;
\forall \mathbf{x}$$

This is a purely theoritecal construct, where the width of the hidden layer of
the model ($\text{dim}(\mathbf{h})$) can grow without bounds. Universal
approximation can also be proved for models having finite width but possibly
infinite depth (as in number of fully-connected layers).
