# Approximating Stochastic Data Sets

## Introduction

Neural networks are [universal function approximators][UAT]. Which means that having enough hidden neurons a neural network can be used to approximate any continuous function. Real life data, however, often has noise or hidden variables which makes approximation inaccurate and in other cases over trained. At best, the prediction settles on the mean of the immediate vicinity. In **Fig. 1** we can see that using a neural network to approximate a noisy data set fails to capture all the information.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig1.png" alt="fig1" style="zoom:80%;" /> |
| **Fig. 1** Approximation of <img src="https://render.githubusercontent.com/render/math?math=ax^3 + bx^2 + cx + d"> with added normal noise |




Because of this it is useful to a have a model that instead of producing a single prediction value for a given input value, produces an arbitrary set of predictions mirroring the dataset distribution at that input.

## The Method

Given that the function producing the data has a specific distribution <img src="https://render.githubusercontent.com/render/math?math=Y"> at input <img src="https://render.githubusercontent.com/render/math?math=x \in X"> [^1] we can define the target function, or the function we actually want to approximate, as <img src="https://render.githubusercontent.com/render/math?math=y \sim Y_{x}">. We want to create an algorithm capable of sampling an arbitrary number of data points from <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> [^2] for any given <img src="https://render.githubusercontent.com/render/math?math=x \in X">.

To do this we introduce a secondary input <img src="https://render.githubusercontent.com/render/math?math=z"> that can be sampled from uniformly distributed space <img src="https://render.githubusercontent.com/render/math?math=Z"> [^3] by the algorithm and fed to a deterministic function <img src="https://render.githubusercontent.com/render/math?math=f(x, z) = y"> where <img src="https://render.githubusercontent.com/render/math?math=y"> is a possible value of the target function with probability <img src="https://render.githubusercontent.com/render/math?math=Pr(Z=z)">. That is <img src="https://render.githubusercontent.com/render/math?math=Pr(Y_{x}=f(x, Z=z)) = Pr(Z=z)">.

Or put another way, we want a deterministic function that maps an input <img src="https://render.githubusercontent.com/render/math?math=x">, a random (but uniform) variable <img src="https://render.githubusercontent.com/render/math?math=Z"> to a dependent random variable <img src="https://render.githubusercontent.com/render/math?math=Y_{x}">.

[^1]: <img src="https://render.githubusercontent.com/render/math?math=X"> is the n-dimensional continuous domain of the target function.
[^2]: <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> is a dependent random variable in an n-dimensional continuous space. The probability function must be continuous on <img src="https://render.githubusercontent.com/render/math?math=x">. In this article and the provided source code <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> is assumed to be 1-dimensional.  
[^3]: <img src="https://render.githubusercontent.com/render/math?math=Z"> is a uniformly distributed random variable in an n-dimensional continuous space with a predefined range, however in this article and the provided code <img src="https://render.githubusercontent.com/render/math?math=Z"> is always assumed to be 1-dimensional.


## Model

The proposed model to approximate <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> is an ordinary feed forward neural network that in addition to an input <img src="https://render.githubusercontent.com/render/math?math=x"> also takes an input <img src="https://render.githubusercontent.com/render/math?math=z"> that can be sampled from <img src="https://render.githubusercontent.com/render/math?math=Z">.

```mermaid
graph LR
A(x) --> B(neural network) --> C(y)
D(z-sample) --> B
```

### Overview

At every point <img src="https://render.githubusercontent.com/render/math?math=x \in X"> we want <img src="https://render.githubusercontent.com/render/math?math=f(x, z \sim Z)"> to approximate the <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> distribution to an arbitrary precision. Let's picture <img src="https://render.githubusercontent.com/render/math?math=f(x, z \sim Z)"> and <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> as 2-dimensional (in the case where <img src="https://render.githubusercontent.com/render/math?math=X"> and <img src="https://render.githubusercontent.com/render/math?math=Z"> are both 1-dimensional) pieces of fabric, they can stretch and shrink in different measures at different regions, decreasing or increasing it's density respectively. We want a mechanism that stretches and shrinks <img src="https://render.githubusercontent.com/render/math?math=f(x, z \sim Z)"> in a way that matches the shrinks and stretches in <img src="https://render.githubusercontent.com/render/math?math=Y_{x}">.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig_x2_uniform.gif" alt="const_uniform" style="zoom:66%;" /> |
| **Fig 2** An animation of the training to match <img src="https://render.githubusercontent.com/render/math?math=x^2"> with added uniform noise. |

In **Fig 2** we can see how the trained model output stretches and shrinks little by little on each epoch until it matches target function.

Going on with the stretching and shrinking piece of fabric analogy, we want to put "pins" into this fabric so that we can superimpose it over the underlying fabric that we are trying to match. We will put these pins into fixed points in the fabric but we will move them to different places of the underlying fabric as we train the model. At first we will pin them to random places. As we observe the position of the pins on the underlying fabric relative to the fabric we will move them slightly upwards or downwards to improve the fabric's match on the underlying fabric. Every pin will affect its surroundings in the fabric proportionally to distance from the pin.

We'll start by putting 1 pin in any given latitude of the fabric and at the midpoint longitude across the fabric's height. We'll then make many observations in the underlying fabric at the same latitude, that is we will randomly pick several locations at the vertical line that goes through the selected pin location.

For every observed point, we'll move the pin position in the underlying fabric a small predefined distance downwards if the observed point is below its current position, and we'll move it upwards if it is above it. This means that if there are more observed points above the pin's position in the underlying fabric the total movement will be upwards and vice versa if there are more observed points below it. If we repeat this process enough times, the pin's position in the underlying fabric will settle in a place that divides the observed points by half, that is the same amount of observed points are above it as below it.
|                                                              |
| ------------------------------------------------------------ |
| <img src="images\fig3.gif" alt="fig3" style="zoom:50%;" /> |
| **Fig 3** Moving 1 pins towards observed points until it settles. |

The pin comes to a stable position dividing all data points in half because the amount of movement for every observation is equal for data points above and below. If the predefined distance of movement for observations above is different from the predefined distance of movement for observations below then the pin would settle in a position dividing the data points by a different ratio (different than half). For example, let's try having 2 pins instead of 1, the first one will move 1 distance for observations above it and 0.5 distance for observations below, the second one will do the opposite. After enough iterations the first one should settle at a position that divides the data points by <img src="https://render.githubusercontent.com/render/math?math=1/3"> above and <img src="https://render.githubusercontent.com/render/math?math=2/3"> below while the second pin will divide by <img src="https://render.githubusercontent.com/render/math?math=2/3"> above and <img src="https://render.githubusercontent.com/render/math?math=1/3"> below. This means we'll have <img src="https://render.githubusercontent.com/render/math?math=1/3"> above the first pin, <img src="https://render.githubusercontent.com/render/math?math=1/3"> between both pins and <img src="https://render.githubusercontent.com/render/math?math=1/3"> below the second pin.

|                                                              |
| ------------------------------------------------------------ |
| <img src="images\fig4.gif" alt="fig4" style="zoom:50%;" /> |
| **Fig 4** Moving 2 pins towards observed points until they settle. |

If  a pin divides the observed data points in 2 groups of sizes <img src="https://render.githubusercontent.com/render/math?math=a"> and <img src="https://render.githubusercontent.com/render/math?math=b"> and after training its fixed position settles in the underlying fabric in the <img src="https://render.githubusercontent.com/render/math?math=a/(a+b)"> longitude from the top, we have a single point mapping between the 2 fabrics, that is at this latitude the densities above and below the pin are equal in both pieces of fabric. We can extrapolate this concept and create as many pins as we want in order to create a finer mapping between the 2 pieces of fabric.

### Definition

We'll start by selecting a fixed set of points in <img src="https://render.githubusercontent.com/render/math?math=Z">  of size <img src="https://render.githubusercontent.com/render/math?math=S"> that we will call ***z-samples***. We can define this set as <img src="https://render.githubusercontent.com/render/math?math=z_{samples} = \{z_0, z_{1}, ... , z_{S}\} \in Z s.t. z_{0} < z_{1}, ... , < z_{S}">.

And we define our goals to train <img src="https://render.githubusercontent.com/render/math?math=f"> as:
###### Goal 1
<img src="https://render.githubusercontent.com/render/math?math=">
\forall x \in X \and \forall z' \in z_{samples}: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z)
<img src="https://render.githubusercontent.com/render/math?math=">
In other words, we want that for every <img src="https://render.githubusercontent.com/render/math?math=z"> in <img src="https://render.githubusercontent.com/render/math?math=z_{samples}"> and across the entire <img src="https://render.githubusercontent.com/render/math?math=X"> input space the cumulative distribution functions <img src="https://render.githubusercontent.com/render/math?math=F_{Y_{x}}(f(x, z'))"> and <img src="https://render.githubusercontent.com/render/math?math=F_{Z}(z')"> have the same value. This first goal gives us a discrete finite mapping between the ***z-samples*** set and <img src="https://render.githubusercontent.com/render/math?math=Y_{x}">. Although it doesn't say anything about all the points <img src="https://render.githubusercontent.com/render/math?math=\hat{z}"> in <img src="https://render.githubusercontent.com/render/math?math=Z"> that are not in <img src="https://render.githubusercontent.com/render/math?math=z_{samples}">.

###### Goal 2

<img src="https://render.githubusercontent.com/render/math?math=">
\forall x \in X \and \forall z', z'' \in Z \space s.t. \space z' < z'': f(x, z') < f(x, z'')
<img src="https://render.githubusercontent.com/render/math?math=">

This second goal gives us that for any given <img src="https://render.githubusercontent.com/render/math?math=x"> in <img src="https://render.githubusercontent.com/render/math?math=X"> <img src="https://render.githubusercontent.com/render/math?math=f"> is monotonically increasing function in <img src="https://render.githubusercontent.com/render/math?math=Z">.

Both of these goals will be tested empirically during the testing step of the training algorithm.

If we assume that these goals are met we have that for any point <img src="https://render.githubusercontent.com/render/math?math=z \sim Z"> and with <img src="https://render.githubusercontent.com/render/math?math=z'"> and <img src="https://render.githubusercontent.com/render/math?math=z''"> being the points in the ***z-samples*** set that are immediately smaller and greater respectively we have that:
<img src="https://render.githubusercontent.com/render/math?math=">
\forall x \in X: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z) < \mathbf{Pr(z >= Z)} < Pr(f(x, z'') >= Y_{x}) = Pr(z'' >= Z)
<img src="https://render.githubusercontent.com/render/math?math=">

and
<img src="https://render.githubusercontent.com/render/math?math=">
\forall x \in X: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z) < \mathbf{Pr(f(x, z) >= Y_{x})} < Pr(f(x, z'') >= Y_{x}) = Pr(z'' >= Z)
<img src="https://render.githubusercontent.com/render/math?math=">

From this we have that:
<img src="https://render.githubusercontent.com/render/math?math=">
\forall x \in X: |Pr(z >= Z) - Pr(f(x, z) >= Y_{x})| < Pr(z'' >= Z) - Pr(z' >= Z) = Pr(f(x, z'') >= Y_{x}) - Pr(f(x, z') >= Y_{x})
<img src="https://render.githubusercontent.com/render/math?math=">
What this means is that for any point <img src="https://render.githubusercontent.com/render/math?math=z \sim Z"> the error of <img src="https://render.githubusercontent.com/render/math?math=f">, which an be defined as <img src="https://render.githubusercontent.com/render/math?math=|Pr(z >= Z) - Pr(f(x, z) >= Y_{x})|">, is always smaller than <img src="https://render.githubusercontent.com/render/math?math=Pr(z'' >= Z) - Pr(z' >= Z)"> or <img src="https://render.githubusercontent.com/render/math?math=Pr(f(x, z'') >= Y_{x}) - Pr(f(x, z') >= Y_{x})"> which (since <img src="https://render.githubusercontent.com/render/math?math=f"> is monotonically increasing in <img src="https://render.githubusercontent.com/render/math?math=z"> for any given <img src="https://render.githubusercontent.com/render/math?math=x"> ) can be minimized by making <img src="https://render.githubusercontent.com/render/math?math=z'' - z'"> smaller. This can be achieved by increasing the number of ***z-samples*** or <img src="https://render.githubusercontent.com/render/math?math=S">. In other words the maximum error of <img src="https://render.githubusercontent.com/render/math?math=f"> can be arbitrarily minimized by a sufficiently large <img src="https://render.githubusercontent.com/render/math?math=S">.
<img src="https://render.githubusercontent.com/render/math?math=">
\exists z_{samples} \subset Z, \forall \epsilon : |Pr(z >= Z) - Pr(f(x, z) >= Y_{x})| < \epsilon
<img src="https://render.githubusercontent.com/render/math?math=">


Having defined our goals and what will they buy us, we move to show how we will achieve [Goal 1](#Goal-1). For simplicity we will use a ***z-samples*** set that is evenly distributed in <img src="https://render.githubusercontent.com/render/math?math=Z">, that is: <img src="https://render.githubusercontent.com/render/math?math=\{z_0, z_{1}, ... , z_{S - 1}\} \in Z s.t. z_{0} < z_{1}, ... , < z_{S} \and Pr(z_{1} > Z > z_{0}) = Pr(z_{2} > Z > z_{1}), ... , Pr(z_{s} > Z > z_{s-1})">.

For any given <img src="https://render.githubusercontent.com/render/math?math=x"> in <img src="https://render.githubusercontent.com/render/math?math=X"> and any <img src="https://render.githubusercontent.com/render/math?math=z'"> in <img src="https://render.githubusercontent.com/render/math?math=z_{samples}"> we want <img src="https://render.githubusercontent.com/render/math?math=f"> to satisfy <img src="https://render.githubusercontent.com/render/math?math=Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z)">. For this purpose we'll assume that we count with a sufficiently representative set of samples in <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> or <img src="https://render.githubusercontent.com/render/math?math=y_{train} \sim Y_{x}">.

For a given <img src="https://render.githubusercontent.com/render/math?math=\bar{x} \in X"> if <img src="https://render.githubusercontent.com/render/math?math=z'"> was the midpoint in <img src="https://render.githubusercontent.com/render/math?math=Z"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=z' \in Z s.t. Pr(z' > Z) = 0.5">) we could simply train <img src="https://render.githubusercontent.com/render/math?math=f"> to change the value of <img src="https://render.githubusercontent.com/render/math?math=f(\bar{x}, z')"> a constant movement number <img src="https://render.githubusercontent.com/render/math?math=M"> greater for every training example <img src="https://render.githubusercontent.com/render/math?math=y \in y_{train}"> that was greater than <img src="https://render.githubusercontent.com/render/math?math=f(\bar{x}, z')"> itself and the same constant number smaller for every training example that was smaller. This would cause after enough iterations for the value of <img src="https://render.githubusercontent.com/render/math?math=f(\bar{x},z')"> to settle in a position that divides in half all training examples when the total movement equals 0.

If instead of being <img src="https://render.githubusercontent.com/render/math?math=Z">'s midpoint <img src="https://render.githubusercontent.com/render/math?math=Pr(z' > Z) \ne 0.5"> then the constant numbers of movement for greater and smaller samples have to be different.

Let's say that <img src="https://render.githubusercontent.com/render/math?math=a"> is the distance between <img src="https://render.githubusercontent.com/render/math?math=z'"> and the smallest number in <img src="https://render.githubusercontent.com/render/math?math=Z"> or <img src="https://render.githubusercontent.com/render/math?math=Z_{min}">, and <img src="https://render.githubusercontent.com/render/math?math=b"> the distance between <img src="https://render.githubusercontent.com/render/math?math=z'"> and <img src="https://render.githubusercontent.com/render/math?math=Z_{max}">.
<img src="https://render.githubusercontent.com/render/math?math=">
a = z' - Z_{min}
<img src="https://render.githubusercontent.com/render/math?math=">

<img src="https://render.githubusercontent.com/render/math?math=">
b = Z_{max} - z'
<img src="https://render.githubusercontent.com/render/math?math=">



Since <img src="https://render.githubusercontent.com/render/math?math=a"> represents the amount of training examples we hope to find smaller than <img src="https://render.githubusercontent.com/render/math?math=z'"> and <img src="https://render.githubusercontent.com/render/math?math=b"> the amount of training examples greater than <img src="https://render.githubusercontent.com/render/math?math=z'"> we need 2 scalars <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and <img src="https://render.githubusercontent.com/render/math?math=\beta"> to satisfy the following equations:
<img src="https://render.githubusercontent.com/render/math?math=">
\alpha a = \beta b
<img src="https://render.githubusercontent.com/render/math?math=">
These scalars will be the multipliers to be used with the constant movement <img src="https://render.githubusercontent.com/render/math?math=M"> on every observed point smaller and greater than <img src="https://render.githubusercontent.com/render/math?math=z'"> respectively. This first equation assures that the total movement when <img src="https://render.githubusercontent.com/render/math?math=z'"> is situated at <img src="https://render.githubusercontent.com/render/math?math=Z_{min} + a"> or <img src="https://render.githubusercontent.com/render/math?math=Z_{max} - b"> will be 0.
<img src="https://render.githubusercontent.com/render/math?math=">
\alpha a + \beta b = 1
<img src="https://render.githubusercontent.com/render/math?math=">
This second equation normalizes the scalars so that the total movement for all <img src="https://render.githubusercontent.com/render/math?math=z"> in <img src="https://render.githubusercontent.com/render/math?math=z_{samples}"> have the same total movement.

Which gives us:
<img src="https://render.githubusercontent.com/render/math?math=">
\alpha = 1 / (2  a)
<img src="https://render.githubusercontent.com/render/math?math=">

<img src="https://render.githubusercontent.com/render/math?math=">
\beta = 1 / (2 b)
<img src="https://render.githubusercontent.com/render/math?math=">



This logic however, breaks at the edges, that is when a *z-sample* is equal to <img src="https://render.githubusercontent.com/render/math?math=Z_{min}"> or <img src="https://render.githubusercontent.com/render/math?math=Z_{max}">. At these values either <img src="https://render.githubusercontent.com/render/math?math=a"> or <img src="https://render.githubusercontent.com/render/math?math=b"> is 0 and if either of them is 0 then one of <img src="https://render.githubusercontent.com/render/math?math=\alpha"> or <img src="https://render.githubusercontent.com/render/math?math=\beta"> is undefined.

As <img src="https://render.githubusercontent.com/render/math?math=a"> or <img src="https://render.githubusercontent.com/render/math?math=b"> approach 0 <img src="https://render.githubusercontent.com/render/math?math=\alpha"> or <img src="https://render.githubusercontent.com/render/math?math=\beta"> tend to infinity, one might be tempted to replace this with a large number, but that would not be practical because a large distance multiplier would dominate the training and minimize the movement of other <img src="https://render.githubusercontent.com/render/math?math=z_{samples}">.

Also as one of <img src="https://render.githubusercontent.com/render/math?math=\alpha"> or <img src="https://render.githubusercontent.com/render/math?math=\beta"> tend to infinity the other one becomes a small number that is also impractical but for a different reason. The <img src="https://render.githubusercontent.com/render/math?math=z_{samples}"> at the edges are supposed to map to the edges of <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> and any quantity of movement into the opposite direction will result in <img src="https://render.githubusercontent.com/render/math?math=Z_{min}"> or <img src="https://render.githubusercontent.com/render/math?math=Z_{max}"> mapping to a greater or smaller point in <img src="https://render.githubusercontent.com/render/math?math=Y_x"> respectively. For this reason the <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and <img src="https://render.githubusercontent.com/render/math?math=\beta"> for the <img src="https://render.githubusercontent.com/render/math?math=z_{samples}"> at the edges (i.e. <img src="https://render.githubusercontent.com/render/math?math=z_{0}"> and <img src="https://render.githubusercontent.com/render/math?math=z_{S}">) will be assigned a value of 0 for the one that pushes inward and a predefined constant <img src="https://render.githubusercontent.com/render/math?math=C \in [0, 1]"> that can be adjusted to the model.

## Training the model

In order to train the neural network the *z-samples* set size <img src="https://render.githubusercontent.com/render/math?math=S"> is chosen depending on the desired accuracy and compute available. Having decided the training level <img src="https://render.githubusercontent.com/render/math?math=Z"> must be defined. That is, the number of dimensions and its range must be chosen. Given <img src="https://render.githubusercontent.com/render/math?math=Z"> and the training level we can create the *z-samples* set.

For example if <img src="https://render.githubusercontent.com/render/math?math=Z"> is 1-dimensional with range defined as <img src="https://render.githubusercontent.com/render/math?math=[10.0, 20.0]"> and <img src="https://render.githubusercontent.com/render/math?math=S"> 9, we have that the *z-sample* set is <img src="https://render.githubusercontent.com/render/math?math=\{z_{0} (10.0), z_{1} (11.25), z_{2} (12.5), z_{3} (13.75), z_{4} (15.0), z_{5} (16.25), z_{6} (17.5), z_{7} (18.75), z_{8} (20.0)\}">.

First we select a batch of data from the training data with size <img src="https://render.githubusercontent.com/render/math?math=n">, for every data point in the batch, we evaluate the current model on every *z-sample*. This gives us the prediction matrix:


<img src="https://render.githubusercontent.com/render/math?math=">
\begin{bmatrix}
f(x_{0}, z_{0}), f(x_{1}, z_{0}), ..., f(x_{0}, z_{S})\\
f(x_{1}, z_{0}), f(x_{1}, z_{1}), ..., f(x_{1}, z_{S})\\
...\\
f(x_{n}, z_{0}), f(x_{n}, z_{1}), ..., f(x_{n}, z_{S})\\
\end{bmatrix}
<img src="https://render.githubusercontent.com/render/math?math=">

For every data point <img src="https://render.githubusercontent.com/render/math?math=(x_{i}, y_{i})"> in the batch, we take the output value <img src="https://render.githubusercontent.com/render/math?math=y_{i}"> and compare it with every value of its corresponding row in the prediction matrix (i.e. <img src="https://render.githubusercontent.com/render/math?math=[f(x_{i}, z_{0}), f(x_{i}, z_{0}), ..., f(x_{i}, z_{8})]">). After determining if <img src="https://render.githubusercontent.com/render/math?math=y_{i}"> is greater or smaller than each predicted value, we produce 2 values for every element in the matrix:

###### Weight
The weight is the scalar <img src="https://render.githubusercontent.com/render/math?math=\alpha_{z-sample}"> if <img src="https://render.githubusercontent.com/render/math?math=y_{i}"> is smaller than the prediction and <img src="https://render.githubusercontent.com/render/math?math=\beta_{z-sample}"> if <img src="https://render.githubusercontent.com/render/math?math=y_{i}"> is greater.


<img src="https://render.githubusercontent.com/render/math?math=">
w_{i, z-sample} = \left\{\begin{array}{11}
\alpha_{z-sample} & y_{i} < f(x_{i}, z_{z-sample})\\
\beta_{z-sample} & y_{i} > f(x_{i}, z_{z-sample})\\
\end{array}\right.
<img src="https://render.githubusercontent.com/render/math?math=">

###### Target Value
The target value is the prediction itself plus the preselected movement constant <img src="https://render.githubusercontent.com/render/math?math=M"> multiplied by -1 if <img src="https://render.githubusercontent.com/render/math?math=y_{i}"> is smaller than the prediction and 1 if <img src="https://render.githubusercontent.com/render/math?math=y_{i}"> is greater. You can think of target values as the "where we want the prediction to be" value.
<img src="https://render.githubusercontent.com/render/math?math=">
t_{i, z-sample} = \left\{\begin{array}{11}
f(x_{i}, z_{z-sample}) + M & y_{i} < f(x_{i}, z_{z-sample})\\
f(x_{i}, z_{z-sample}) - M & y_{i} > f(x_{i}, z_{z-sample})\\
\end{array}\right.
<img src="https://render.githubusercontent.com/render/math?math=">


After calculating these 2 values we are ready to assemble the matrix to be used during backpropagation.
<img src="https://render.githubusercontent.com/render/math?math=">
\begin{bmatrix}
(w_{0,0}, t_{0,0}), (w_{0,1}, t_{0,1}), ... , (w_{0,S}, t_{0,S})\\
(w_{1,0}, t_{1,0}), (w_{1,1}, t_{1,1}), ... , (w_{1,S}, t_{1,S})\\
...\\
(w_{n,0}, t_{n,0}), (w_{n,1}, t_{n,1}), ... , (w_{n,S}, t_{n,S})\\
\end{bmatrix}
<img src="https://render.githubusercontent.com/render/math?math=">

We pass the prediction matrix results in a addition to this matrix to a Weighted Mean Squared Error loss function (WMSE). The loss will look like this:
<img src="https://render.githubusercontent.com/render/math?math=">
\sum_{j=0}^{S}\sum_{i=0}^{n}(f(x_{i}, z_{j}) - t_{i,j})^2 * w_{i,j}
<img src="https://render.githubusercontent.com/render/math?math=">



## Testing the model

The Mean Squared Error (MSE) loss function works to train the model using backpropagation and target values, but testing the model requires a different approach. Since the <img src="https://render.githubusercontent.com/render/math?math=Z"> and <img src="https://render.githubusercontent.com/render/math?math=Y_{x}"> are random, measuring the differences between them is pointless. Because of this, the success of the model will be measured in 2 ways:

### Earth Movers Distance (EMD)

> In statistics, the **earth mover's distance** (**EMD**) is a measure of the distance between two probability distributions over a region *D*. In mathematics, this is known as the Wasserstein metric. Informally, if the distributions are interpreted as two different ways of piling up a certain amount of dirt over the region *D*, the EMD is the minimum cost of turning one pile into the other; where the cost is assumed to be amount of dirt moved times the distance by which it is moved. [wikipedia.org][EMD]

Using EMD we can obtain an indicator of how similar <img src="https://render.githubusercontent.com/render/math?math=\forall x \in X: y \sim Y_{x}"> and <img src="https://render.githubusercontent.com/render/math?math=\forall x \in X : f(x, z \sim Z)"> are. It can be calculated by comparing every <img src="https://render.githubusercontent.com/render/math?math=(x, y)"> data point in the test data and prediction data sets and finding way to transform one into the other that requires the smallest total movement.

|      |
| ---- |
|   <img src="images\fig5.png" alt="fig5" style="zoom:50%;" />   |
| **Fig 5** EMD testing. |

### Testing the training goals
#### Training goal 1

Ideally to test [Goal 1](#Goal-1) (i.e. <img src="https://render.githubusercontent.com/render/math?math=\forall x \in X \and \forall z' \in z_{samples}: Pr(f(x, z') >= Y_{x}) = Pr(z' >= Z)">) we would evaluate <img src="https://render.githubusercontent.com/render/math?math=f"> for a given <img src="https://render.githubusercontent.com/render/math?math=x"> and on every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> and then compare it to an arbitrary number of test data points having the same <img src="https://render.githubusercontent.com/render/math?math=x">. We would then proceed to count for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> the number of test data points smaller than it. With a vector of *smaller than counts* (i.e. <img src="https://render.githubusercontent.com/render/math?math=Pr(f(x, z') >= Y_{x})">) we could proceed to compare it with the canonical counts for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=Pr(z' >= Z)">) and measure the error. In real life data sets this is not possible however. Real life data sets will not likely have an arbitrary number of data points having the same <img src="https://render.githubusercontent.com/render/math?math=x"> (they will unlikely have 2 data points with the same <img src="https://render.githubusercontent.com/render/math?math=x">) which means that we need to use a vicinity in <img src="https://render.githubusercontent.com/render/math?math=x"> (values <img src="https://render.githubusercontent.com/render/math?math=X"> that are close to an <img src="https://render.githubusercontent.com/render/math?math=x">) to test the goal.

We start by creating an ordering (an array of indices) <img src="https://render.githubusercontent.com/render/math?math=O_{d} = \{o_{0}, o_{1}, ..., o_{m}\}"> that sorts all the elements in <img src="https://render.githubusercontent.com/render/math?math=X_{test}"> (all the <img src="https://render.githubusercontent.com/render/math?math=x"> inputs in the test data set) on dimension <img src="https://render.githubusercontent.com/render/math?math=d">. Then we select an element in <img src="https://render.githubusercontent.com/render/math?math=O_{d}"> and pick the <img src="https://render.githubusercontent.com/render/math?math=V"> (vicinity size) samples closest to it. This gives us a subset of  consecutive elements in <img src="https://render.githubusercontent.com/render/math?math=O_{d}"> which we'll call <img src="https://render.githubusercontent.com/render/math?math=G = \{o_{0}, o_{1}, ... , o_{V}\}">.

Now we can evaluate <img src="https://render.githubusercontent.com/render/math?math=f"> for every <img src="https://render.githubusercontent.com/render/math?math=x_{o'} \mid o' \in G"> on every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> which gives us the matrix:
<img src="https://render.githubusercontent.com/render/math?math=">
\begin{bmatrix}
f(x_{o_{0}}, z_{0}), f(x_{o_{0}}, z_{1}), ... , f(x_{o_{0}}, z_{S})\\
f(x_{o_{1}}, z_{0}), f(x_{o_{1}}, z_{1}), ... , f(x_{o_{1}}, z_{S})\\
...\\
f(x_{o_{V}}, z_{0}), f(x_{o_{V}}, z_{1}), ... , f(x_{o_{V}}, z_{S})\\
\end{bmatrix}
<img src="https://render.githubusercontent.com/render/math?math=">
We then proceed to compare each row with the outputs <img src="https://render.githubusercontent.com/render/math?math=y_{o'} \mid o' \in G">
<img src="https://render.githubusercontent.com/render/math?math=">
\begin{bmatrix}
y_{o_{0}}\\
y_{o_{1}}\\
...\\
y_{o_{V}}\\
\end{bmatrix}
<img src="https://render.githubusercontent.com/render/math?math=">

and create *smaller than counts* (i.e. <img src="https://render.githubusercontent.com/render/math?math=Pr(f(x, z) >= Y_{x})">) which we can then compare with the canonical counts for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=Pr(z >= Z)">) to measure the error in the selected vicinity.

We will create a number of such vicinities and call each error as the local errors and a vicinity covering all <img src="https://render.githubusercontent.com/render/math?math=X_{test}"> and call its error the mean error.

|      |
| ---- |
|   <img src="images\fig6.png" alt="fig6" style="zoom:50%;" />   |
| **Fig 6** Training goal 1 testing. |



#### Training goal 2

In order to test [Goal 2](#Goal-2) <img src="https://render.githubusercontent.com/render/math?math=\forall x \in X \and \forall z', z'' \in z_{samples} \space s.t. \space z' < z'': f(x, z') < f(x, z'')"> we select some random points in <img src="https://render.githubusercontent.com/render/math?math=X"> and a set of random points in <img src="https://render.githubusercontent.com/render/math?math=Z">, we run them in our model and get result matrix:
<img src="https://render.githubusercontent.com/render/math?math=">
\begin{bmatrix}
f(x_{0}, z_{0}), f(x_{0}, z_{1}), ... , f(x_{0}, z_{n})\\
f(x_{1}, z_{0}), f(x_{1}, z_{1}), ... , f(x_{1}, z_{n})\\
...\\
f(x_{m}, z_{0}), f(x_{m}, z_{1}), ... , f(x_{m}, z_{n})\\
\end{bmatrix}
<img src="https://render.githubusercontent.com/render/math?math=">


From here it is trivial to check that each row is monotonically increasing. To increase the quality of the check we can increment the sizes of the test point set in <img src="https://render.githubusercontent.com/render/math?math=X"> and the test point set in <img src="https://render.githubusercontent.com/render/math?math=Z">.

## Experiments

The following are various experiments done on different datasets.

### <img src="https://render.githubusercontent.com/render/math?math=x^2"> plus gaussian noise

Let's start with a simple example. The function <img src="https://render.githubusercontent.com/render/math?math=x_{2}"> with added gaussian noise. On the left panel you can see the training evolving over the course of 180 epochs. On the top left corner of this panel you can see the goal 1 error localized over <img src="https://render.githubusercontent.com/render/math?math=X">, at the end of the training you can see that the highest local error is around 2% and the global error is around 0.5%. On the top right corner of the same panel you can see the local Earth Mover's Distance (EMD). On the bottom left corner you can see a plot of the original test dataset (in blue) and the <img src="https://render.githubusercontent.com/render/math?math=z-lines"> (in orange), you can see how they represent the predicted space's mapping on the real data. On the bottom right you can see a plot of the original test dataset (in blue) and random predictions (with <img src="https://render.githubusercontent.com/render/math?math=z \sim Z">), you can see as the predicted results slowly conform to the real data.

On the right panel, you can see a plot of the global goal1 error (above) and global EMD values (below) as they change over the entire training.

|      |      |
| ---- | ---- |
|   <img src="images\x2_normal_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_x2norm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 7** Training goal 1 testing. |  |

### <img src="https://render.githubusercontent.com/render/math?math=ax^3 + bx^2 + cx + d"> plus truncated gaussian noise

This one is a bit more complicated. An order 3 polynomial with added truncated gaussian noise, that is a normal distribution clipped at specific points.

|      |      |
| ---- | ---- |
|   <img src="images\x3x2_trunc_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_x3x2trunc_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 8** Training goal 1 testing. |  |

### Double <img src="https://render.githubusercontent.com/render/math?math=sin(x)"> plus gaussian noise multiplied by sin(x)

This one is quite more interesting. 2 mirroring <img src="https://render.githubusercontent.com/render/math?math=sin(x)"> functions with gaussian noise scaled by <img src="https://render.githubusercontent.com/render/math?math=sin(x)"> itself.
<img src="https://render.githubusercontent.com/render/math?math=">
f = \left\{\begin{array}{11}
sin(x) + \mathcal{N} * sin(x) & U(0,1) <= 0.5\\
-sin(x) + \mathcal{N} * sin(x) & U(0,1) > 0.5\\
\end{array}\right.
<img src="https://render.githubusercontent.com/render/math?math=">


|      |      |
| ---- | ---- |
|   <img src="images\sin_sin_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_sinsin_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| \ |  |

### Branching function plus gaussian noise

This one adds branching.

|      |      |
| ---- | ---- |
|   <img src="images\branch_norm_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_branchnorm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 10** Training goal 1 testing. |  |

### <img src="https://render.githubusercontent.com/render/math?math=x_{2}"> in one dimension <img src="https://render.githubusercontent.com/render/math?math=x_{3}"> in another plus absolute normal

The next example has 2 dimensions of input. <img src="https://render.githubusercontent.com/render/math?math=X_{0}"> (the first dimension) is <img src="https://render.githubusercontent.com/render/math?math=x^2"> and <img src="https://render.githubusercontent.com/render/math?math=X_{1}"> (the second dimension) is <img src="https://render.githubusercontent.com/render/math?math=x^3">.

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
|<img src="images\california_housing_plots_res_0.gif" alt="fig10_0" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_1.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_2.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_3.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_4.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_5.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_6.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_7.gif" alt="fig10_1" style="zoom:66%;" />|
|<img src="images\fig_cal_tensorboard.png" alt="fig7" style="zoom:50%;" />|
| **Fig 10** Training goal 1 testing. |

## Conclusion

