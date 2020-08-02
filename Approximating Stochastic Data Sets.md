# Approximating Stochastic Data Sets

## Introduction

Neural networks are [universal function approximators][UAT]. Which means that having enough hidden neurons a neural network can be used to approximate any continuous function. Real life data, however, often has noise or hidden variables which makes approximation inaccurate and in other cases over trained. At best, the prediction settles on the mean of the immediate vicinity. In **Fig. 1** we can see that using a neural network to approximate a noisy data set fails to capture all the information.

|                                                              |
| :----------------------------------------------------------: |
| <img src="D:\research\sffnn\images\fig1.png" alt="fig1" style="zoom:80%;" /> |
| **Fig. 1** Approximation of $ax^3 + bx^2 + cx + d$ with added normal noise |




Because of this it is useful to a have a model that instead of producing a single prediction value for a given input value, produces an arbitrary set of predictions mirroring the dataset distribution at that input.

## The Method

Given that the function producing the data has a specific distribution $Y$ at input $x \in X$ [^1] we can define the target function, or the function we actually want to approximate, as $y \sim Y_{x}$. We want to create an algorithm capable of sampling an arbitrary number of data points from $Y_{x}$ [^2] for any given $x \in X$.

To do this we introduce a secondary input $z$ that can be sampled from uniformly distributed space $Z$ [^3] by the algorithm and fed to a deterministic function $f(x, z) = y$ where $y$ is a possible value of the target function with probability $Pr(Z=z)$. That is $Pr(Y_{x}=f(x, Z=z)) = Pr(Z=z)$.

Or put another way, we want a deterministic function that maps an input $x$, a random (but uniform) variable $Z$ to a dependent random variable $Y_{x}$.

[^1]: $X$ is the n-dimensional continuous domain of the target function.
[^2]: $Y_{x}$ is a dependent random variable in an n-dimensional continuous space. The probability function must be continuous on $x$. In this article and the provided source code $Y_{x}$ is assumed to be 1-dimensional.  
[^3]: $Z$ is a uniformly distributed random variable in an n-dimensional continuous space with a predefined range, however in this article and the provided code $Z$ is always assumed to be 1-dimensional.


## Model

The proposed model to approximate $Y_{x}$ is an ordinary feed forward neural network that in addition to an input $x$ also takes an input $z$ that can be sampled from $Z$.

```mermaid
graph LR
A(x) --> B(neural network) --> C(y)
D(z-sample) --> B
```

### Overview

At every point $x \in X$ we want $f(x, z \sim Z)$ to approximate the $Y_{x}$ distribution to an arbitrary precision. Let's picture $f(x, z \sim Z)$ and $Y_{x}$ as 2-dimensional (in the case where $X$ and $Z$ are both 1-dimensional) pieces of fabric, they can stretch and shrink in different measures at different regions, decreasing or increasing it's density respectively. We want a mechanism that stretches and shrinks $f(x, z \sim Z)$ in a way that matches the shrinks and stretches in $Y_{x}$.

|                                                              |
| :----------------------------------------------------------: |
| <img src="D:\research\sffnn\images\fig_x2_uniform.gif" alt="const_uniform" style="zoom:66%;" /> |
| **Fig 2** An animation of the training to match $x^2$ with added uniform noise. |

In **Fig 2** we can see how the trained model output stretches and shrinks little by little on each epoch until it matches target function.

Going on with the stretching and shrinking piece of fabric analogy, we want to put "pins" into this fabric so that we can superimpose it over the underlying fabric that we are trying to match. We will put these pins into fixed points in the fabric but we will move them to different places of the underlying fabric as we train the model. At first we will pin them to random places. As we observe the position of the pins on the underlying fabric relative to the fabric we will move them slightly upwards or downwards to improve the fabric's match on the underlying fabric. Every pin will affect its surroundings in the fabric proportionally to distance from the pin.

We'll start by putting 1 pin in any given latitude of the fabric and at the midpoint longitude across the fabric's height. We'll then make many observations in the underlying fabric at the same latitude, that is we will randomly pick several locations at the vertical line that goes through the selected pin location.

For every observed point, we'll move the pin position in the underlying fabric a small predefined distance downwards if the observed point is below its current position, and we'll move it upwards if it is above it. This means that if there are more observed points above the pin's position in the underlying fabric the total movement will be upwards and vice versa if there are more observed points below it. If we repeat this process enough times, the pin's position in the underlying fabric will settle in a place that divides the observed points by half, that is the same amount of observed points are above it as below it.
|                                                              |
| ------------------------------------------------------------ |
| <img src="D:\research\sffnn\images\fig3.gif" alt="fig3" style="zoom:50%;" /> |
| **Fig 3** Moving 1 pins towards observed points until it settles. |

The pin comes to a stable position dividing all data points in half because the amount of movement for every observation is equal for data points above and below. If the predefined distance of movement for observations above is different from the predefined distance of movement for observations below then the pin would settle in a position dividing the data points by a different ratio (different than half). For example, let's try having 2 pins instead of 1, the first one will move 1 distance for observations above it and 0.5 distance for observations below, the second one will do the opposite. After enough iterations the first one should settle at a position that divides the data points by $1/3$ above and $2/3$ below while the second pin will divide by $2/3$ above and $1/3$ below. This means we'll have $1/3$ above the first pin, $1/3$ between both pins and $1/3$ below the second pin.

|                                                              |
| ------------------------------------------------------------ |
| <img src="D:\research\sffnn\images\fig4.gif" alt="fig4" style="zoom:50%;" /> |
| **Fig 4** Moving 2 pins towards observed points until they settle. |

If  a pin divides the observed data points in 2 groups of sizes $a$ and $b$ and after training its fixed position settles in the underlying fabric in the $a/(a+b)$ longitude from the top, we have a single point mapping between the 2 fabrics, that is at this latitude the densities above and below the pin are equal in both pieces of fabric. We can extrapolate this concept and create as many pins as we want in order to create a finer mapping between the 2 pieces of fabric.

### Definition

We'll start by selecting a fixed set of points in $Z$  of size $S$ that we will call ***z-samples***. We can define this set as $z_{samples} = \{z_0, z_{1}, ... , z_{S}\} \in Z s.t. z_{0} < z_{1}, ... , < z_{S}$.

And we define our goals to train $f$ as:
###### Goal 1
$$
\forall x \in X \and \forall z' \in z_{samples}: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z)
$$
In other words, we want that for every $z$ in $z_{samples}$ and across the entire $X$ input space the cumulative distribution functions $F_{Y_{x}}(f(x, z'))$ and $F_{Z}(z')$ have the same value. This first goal gives us a discrete finite mapping between the ***z-samples*** set and $Y_{x}$. Although it doesn't say anything about all the points $\hat{z}$ in $Z$ that are not in $z_{samples}$.

###### Goal 2

$$
\forall x \in X \and \forall z', z'' \in Z \space s.t. \space z' < z'': f(x, z') < f(x, z'')
$$

This second goal gives us that for any given $x$ in $X$ $f$ is monotonically increasing function in $Z$.

Both of these goals will be tested empirically during the testing step of the training algorithm.

If we assume that these goals are met we have that for any point $z \sim Z$ and with $z'$ and $z''$ being the points in the ***z-samples*** set that are immediately smaller and greater respectively we have that:
$$
\forall x \in X: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z) < \mathbf{Pr(z >= Z)} < Pr(f(x, z'') >= Y_{x}) = Pr(z'' >= Z)
$$

and
$$
\forall x \in X: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z) < \mathbf{Pr(f(x, z) >= Y_{x})} < Pr(f(x, z'') >= Y_{x}) = Pr(z'' >= Z)
$$

From this we have that:
$$
\forall x \in X: |Pr(z >= Z) - Pr(f(x, z) >= Y_{x})| < Pr(z'' >= Z) - Pr(z' >= Z) = Pr(f(x, z'') >= Y_{x}) - Pr(f(x, z') >= Y_{x})
$$
What this means is that for any point $z \sim Z$ the error of $f$, which an be defined as $|Pr(z >= Z) - Pr(f(x, z) >= Y_{x})|$, is always smaller than $Pr(z'' >= Z) - Pr(z' >= Z)$ or $Pr(f(x, z'') >= Y_{x}) - Pr(f(x, z') >= Y_{x})$ which (since $f$ is monotonically increasing in $z$ for any given $x$ ) can be minimized by making $z'' - z'$ smaller. This can be achieved by increasing the number of ***z-samples*** or $S$. In other words the maximum error of $f$ can be arbitrarily minimized by a sufficiently large $S$.
$$
\exists z_{samples} \subset Z, \forall \epsilon : |Pr(z >= Z) - Pr(f(x, z) >= Y_{x})| < \epsilon
$$


Having defined our goals and what will they buy us, we move to show how we will achieve [Goal 1](#Goal-1). For simplicity we will use a ***z-samples*** set that is evenly distributed in $Z$, that is: $\{z_0, z_{1}, ... , z_{S - 1}\} \in Z s.t. z_{0} < z_{1}, ... , < z_{S} \and Pr(z_{1} > Z > z_{0}) = Pr(z_{2} > Z > z_{1}), ... , Pr(z_{s} > Z > z_{s-1})$.

For any given $x$ in $X$ and any $z'$ in $z_{samples}$ we want $f$ to satisfy $Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z)$. For this purpose we'll assume that we count with a sufficiently representative set of samples in $Y_{x}$ or $y_{train} \sim Y_{x}$.

For a given $\bar{x} \in X$ if $z'$ was the midpoint in $Z$ (i.e. $z' \in Z s.t. Pr(z' > Z) = 0.5$) we could simply train $f$ to change the value of $f(\bar{x}, z')$ a constant movement number $M$ greater for every training example $y \in y_{train}$ that was greater than $f(\bar{x}, z')$ itself and the same constant number smaller for every training example that was smaller. This would cause after enough iterations for the value of $f(\bar{x},z')$ to settle in a position that divides in half all training examples when the total movement equals 0.

If instead of being $Z$'s midpoint $Pr(z' > Z) \ne 0.5$ then the constant numbers of movement for greater and smaller samples have to be different.

Let's say that $a$ is the distance between $z'$ and the smallest number in $Z$ or $Z_{min}$, and $b$ the distance between $z'$ and $Z_{max}$.
$$
a = z' - Z_{min}
$$

$$
b = Z_{max} - z'
$$



Since $a$ represents the amount of training examples we hope to find smaller than $z'$ and $b$ the amount of training examples greater than $z'$ we need 2 scalars $\alpha$ and $\beta$ to satisfy the following equations:
$$
\alpha a = \beta b
$$
These scalars will be the multipliers to be used with the constant movement $M$ on every observed point smaller and greater than $z'$ respectively. This first equation assures that the total movement when $z'$ is situated at $Z_{min} + a$ or $Z_{max} - b$ will be 0.
$$
\alpha a + \beta b = 1
$$
This second equation normalizes the scalars so that the total movement for all $z$ in $z_{samples}$ have the same total movement.

Which gives us:
$$
\alpha = 1 / (2  a)
$$

$$
\beta = 1 / (2 b)
$$



This logic however, breaks at the edges, that is when a *z-sample* is equal to $Z_{min}$ or $Z_{max}$. At these values either $a$ or $b$ is 0 and if either of them is 0 then one of $\alpha$ or $\beta$ is undefined.

As $a$ or $b$ approach 0 $\alpha$ or $\beta$ tend to infinity, one might be tempted to replace this with a large number, but that would not be practical because a large distance multiplier would dominate the training and minimize the movement of other $z_{samples}$.

Also as one of $\alpha$ or $\beta$ tend to infinity the other one becomes a small number that is also impractical but for a different reason. The $z_{samples}$ at the edges are supposed to map to the edges of $Y_{x}$ and any quantity of movement into the opposite direction will result in $Z_{min}$ or $Z_{max}$ mapping to a greater or smaller point in $Y_x$ respectively. For this reason the $\alpha$ and $\beta$ for the $z_{samples}$ at the edges (i.e. $z_{0}$ and $z_{S}$) will be assigned a value of 0 for the one that pushes inward and a predefined constant $C \in [0, 1]$ that can be adjusted to the model.

## Training the model

In order to train the neural network the *z-samples* set size $S$ is chosen depending on the desired accuracy and compute available. Having decided the training level $Z$ must be defined. That is, the number of dimensions and its range must be chosen. Given $Z$ and the training level we can create the *z-samples* set.

For example if $Z$ is 1-dimensional with range defined as $[10.0, 20.0]$ and $S$ 9, we have that the *z-sample* set is $\{z_{0} (10.0), z_{1} (11.25), z_{2} (12.5), z_{3} (13.75), z_{4} (15.0), z_{5} (16.25), z_{6} (17.5), z_{7} (18.75), z_{8} (20.0)\}$.

First we select a batch of data from the training data with size $n$, for every data point in the batch, we evaluate the current model on every *z-sample*. This gives us the prediction matrix:


$$
\begin{bmatrix}
f(x_{0}, z_{0}), f(x_{1}, z_{0}), ..., f(x_{0}, z_{S})\\
f(x_{1}, z_{0}), f(x_{1}, z_{1}), ..., f(x_{1}, z_{S})\\
...\\
f(x_{n}, z_{0}), f(x_{n}, z_{1}), ..., f(x_{n}, z_{S})\\
\end{bmatrix}
$$

For every data point $(x_{i}, y_{i})$ in the batch, we take the output value $y_{i}$ and compare it with every value of its corresponding row in the prediction matrix (i.e. $[f(x_{i}, z_{0}), f(x_{i}, z_{0}), ..., f(x_{i}, z_{8})]$). After determining if $y_{i}$ is greater or smaller than each predicted value, we produce 2 values for every element in the matrix:

###### Weight
The weight is the scalar $\alpha_{z-sample}$ if $y_{i}$ is smaller than the prediction and $\beta_{z-sample}$ if $y_{i}$ is greater.


$$
w_{i, z-sample} = \left\{\begin{array}{11}
\alpha_{z-sample} & y_{i} < f(x_{i}, z_{z-sample})\\
\beta_{z-sample} & y_{i} > f(x_{i}, z_{z-sample})\\
\end{array}\right.
$$

###### Target Value
The target value is the prediction itself plus the preselected movement constant $M$ multiplied by -1 if $y_{i}$ is smaller than the prediction and 1 if $y_{i}$ is greater. You can think of target values as the "where we want the prediction to be" value.
$$
t_{i, z-sample} = \left\{\begin{array}{11}
f(x_{i}, z_{z-sample}) + M & y_{i} < f(x_{i}, z_{z-sample})\\
f(x_{i}, z_{z-sample}) - M & y_{i} > f(x_{i}, z_{z-sample})\\
\end{array}\right.
$$


After calculating these 2 values we are ready to assemble the matrix to be used during backpropagation.
$$
\begin{bmatrix}
(w_{0,0}, t_{0,0}), (w_{0,1}, t_{0,1}), ... , (w_{0,S}, t_{0,S})\\
(w_{1,0}, t_{1,0}), (w_{1,1}, t_{1,1}), ... , (w_{1,S}, t_{1,S})\\
...\\
(w_{n,0}, t_{n,0}), (w_{n,1}, t_{n,1}), ... , (w_{n,S}, t_{n,S})\\
\end{bmatrix}
$$

We pass the prediction matrix results in a addition to this matrix to a Weighted Mean Squared Error loss function (WMSE). The loss will look like this:
$$
\sum_{j=0}^{S}\sum_{i=0}^{n}(f(x_{i}, z_{j}) - t_{i,j})^2 * w_{i,j}
$$



## Testing the model

The Mean Squared Error (MSE) loss function works to train the model using backpropagation and target values, but testing the model requires a different approach. Since the $Z$ and $Y_{x}$ are random, measuring the differences between them is pointless. Because of this, the success of the model will be measured in 2 ways:

### Earth Movers Distance (EMD)

> In statistics, the **earth mover's distance** (**EMD**) is a measure of the distance between two probability distributions over a region *D*. In mathematics, this is known as the Wasserstein metric. Informally, if the distributions are interpreted as two different ways of piling up a certain amount of dirt over the region *D*, the EMD is the minimum cost of turning one pile into the other; where the cost is assumed to be amount of dirt moved times the distance by which it is moved. [wikipedia.org][EMD]

Using EMD we can obtain an indicator of how similar $\forall x \in X: y \sim Y_{x}$ and $\forall x \in X : f(x, z \sim Z)$ are. It can be calculated by comparing every $(x, y)$ data point in the test data and prediction data sets and finding way to transform one into the other that requires the smallest total movement.

|      |
| ---- |
|   <img src="D:\research\sffnn\images\fig5.png" alt="fig5" style="zoom:50%;" />   |
| **Fig 5** EMD testing. |

### Testing the training goals
#### Training goal 1

Ideally to test [Goal 1](#Goal-1) (i.e. $\forall x \in X \and \forall z' \in z_{samples}: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z)$) we would evaluate $f$ for a given $x$ and on every $z-sample$ and then compare it to an arbitrary number of test data points having the same $x$. We would then proceed to count for every $z-sample$ the number of test data points smaller than it. With a vector of *smaller than counts* (i.e. $Pr(f(x, z') >= Y_{x})$) we could proceed to compare it with the canonical counts for every $z-sample$ (i.e. $Pr(z' >= Z)$) and measure the error. In real life data sets this is not possible however. Real life data sets will not likely have an arbitrary number of data points having the same $x$ (they will unlikely have 2 data points with the same $x$) which means that we need to use a vicinity in $x$ (values $X$ that are close to an $x$) to test the goal.

We start by creating an ordering (an array of indices) $O_{d} = \{o_{0}, o_{1}, ..., o_{m}\}$ that sorts all the elements in $X_{test}$ (all the $x$ inputs in the test data set) on dimension $d$. Then we select an element in $O_{d}$ and pick the $V$ (vicinity size) samples closest to it. This gives us a subset of  consecutive elements in $O_{d}$ which we'll call $G = \{o_{0}, o_{1}, ... , o_{V}\}$.

Now we can evaluate $f$ for every $x_{o'} \mid o' \in G$ on every $z-sample$ which gives us the matrix:
$$
\begin{bmatrix}
f(x_{o_{0}}, z_{0}), f(x_{o_{0}}, z_{1}), ... , f(x_{o_{0}}, z_{S})\\
f(x_{o_{1}}, z_{0}), f(x_{o_{1}}, z_{1}), ... , f(x_{o_{1}}, z_{S})\\
...\\
f(x_{o_{V}}, z_{0}), f(x_{o_{V}}, z_{1}), ... , f(x_{o_{V}}, z_{S})\\
\end{bmatrix}
$$
We then proceed to compare each row with the outputs $y_{o'} \mid o' \in G$
$$
\begin{bmatrix}
y_{o_{0}}\\
y_{o_{1}}\\
...\\
y_{o_{V}}\\
\end{bmatrix}
$$

and create *smaller than counts* (i.e. $Pr(f(x, z) >= Y_{x})$) which we can then compare with the canonical counts for every $z-sample$ (i.e. $Pr(z >= Z)$) to measure the error in the selected vicinity.

We will create a number of such vicinities and call each error as the local errors and a vicinity covering all $X_{test}$ and call its error the mean error.

|      |
| ---- |
|   <img src="D:\research\sffnn\images\fig6.png" alt="fig6" style="zoom:50%;" />   |
| **Fig 6** Training goal 1 testing. |



#### Training goal 2

In order to test [Goal 2](#Goal-2) $\forall x \in X \and \forall z', z'' \in z_{samples} \space s.t. \space z' < z'': f(x, z') < f(x, z'')$ we select some random points in $X$ and a set of random points in $Z$, we run them in our model and get result matrix:
$$
\begin{bmatrix}
f(x_{0}, z_{0}), f(x_{0}, z_{1}), ... , f(x_{0}, z_{n})\\
f(x_{1}, z_{0}), f(x_{1}, z_{1}), ... , f(x_{1}, z_{n})\\
...\\
f(x_{m}, z_{0}), f(x_{m}, z_{1}), ... , f(x_{m}, z_{n})\\
\end{bmatrix}
$$


From here it is trivial to check that each row is monotonically increasing. To increase the quality of the check we can increment the sizes of the test point set in $X$ and the test point set in $Z$.

## Experiments

The following are various experiments done on different datasets.

### $x^2$ plus gaussian noise

Let's start with a simple example. The function $x_{2}$ with added gaussian noise. On the left panel you can see the training evolving over the course of 180 epochs. On the top left corner of this panel you can see the goal 1 error localized over $X$, at the end of the training you can see that the highest local error is around 2% and the global error is around 0.5%. On the top right corner of the same panel you can see the local Earth Mover's Distance (EMD). On the bottom left corner you can see a plot of the original test dataset (in blue) and the $z-lines$ (in orange), you can see how they represent the predicted space's mapping on the real data. On the bottom right you can see a plot of the original test dataset (in blue) and random predictions (with $z \sim Z$), you can see as the predicted results slowly conform to the real data.

On the right panel, you can see a plot of the global goal1 error (above) and global EMD values (below) as they change over the entire training.

|      |      |
| ---- | ---- |
|   <img src="D:\research\sffnn\images\x2_normal_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="D:\research\sffnn\images\fig_x2norm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 7** Training goal 1 testing. |  |

### $ax^3 + bx^2 + cx + d$ plus truncated gaussian noise

This one is a bit more complicated. An order 3 polynomial with added truncated gaussian noise, that is a normal distribution clipped at specific points.

|      |      |
| ---- | ---- |
|   <img src="D:\research\sffnn\images\x3x2_trunc_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="D:\research\sffnn\images\fig_x3x2trunc_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 8** Training goal 1 testing. |  |

### Double $sin(x)$ plus gaussian noise multiplied by sin(x)

This one is quite more interesting. 2 mirroring $sin(x)$ functions with gaussian noise scaled by $sin(x)$ itself.
$$
f = \left\{\begin{array}{11}
sin(x) + \mathcal{N} * sin(x) & U(0,1) <= 0.5\\
-sin(x) + \mathcal{N} * sin(x) & U(0,1) > 0.5\\
\end{array}\right.
$$


|      |      |
| ---- | ---- |
|   <img src="D:\research\sffnn\images\sin_sin_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="D:\research\sffnn\images\fig_sinsin_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| \ |  |

### Branching function plus gaussian noise

This one adds branching.

|      |      |
| ---- | ---- |
|   <img src="D:\research\sffnn\images\branch_norm_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="D:\research\sffnn\images\fig_branchnorm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 10** Training goal 1 testing. |  |

### $x_{2}$ in one dimension $x_{3}$ in another plus absolute normal

The next example has 2 dimensions of input. $X_{0}$ (the first dimension) is $x^2$ and $X_{1}$ (the second dimension) is $x^3$.

|      |
| :--: |
|<img src="D:\research\sffnn\images\x3_x2_absnormal_plots_res_0.gif" alt="fig10_0" style="zoom:80%;" />|
|<img src="D:\research\sffnn\images\x3_x2_absnormal_plots_res_1.gif" alt="fig10_1" style="zoom:80%;" />|
|<img src="D:\research\sffnn\images\fig_x3x2abs_tensorboard.png" alt="fig7" style="zoom:50%;" />|
| **Fig 10** Training goal 1 testing. |

### California housing dataset

This is the classic California housing dataset.

|      |
| ---- |
|<img src="D:\research\sffnn\images\california_housing_plots_res_0.gif" alt="fig10_0" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_1.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_2.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_3.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_4.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_5.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_6.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\california_housing_plots_res_7.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="D:\research\sffnn\images\fig_cal_tensorboard.png" alt="fig7" style="zoom:50%;" />|
| **Fig 10** Training goal 1 testing. |

## Conclusion

