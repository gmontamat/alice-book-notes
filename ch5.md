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

If we define $\mathbf{A} = \mathbf{w}_2^{\top} \mathbf{W}_1 \sim (c_1)$ and
$c = \mathbf{w}_2^{\top} \mathbf{b}_1 + b_2 \in \mathbb{R}$, then:

$$\textit{f}_2(\textit{f}_1(\mathbf{x})) = \mathbf{A} \mathbf{x} + c$$

Hence, composing linear models will throw the same results as just using a plain
linear model.
