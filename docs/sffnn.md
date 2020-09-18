# Approximating Stochastic Data Sets

###### Nicolas Arroyo nicolas.arroyo.duran@gmail.com



## Introduction

Neural networks are [universal function approximators][UAT]. Which means that having enough hidden neurons a neural network can be used to approximate any continuous function. Real life data, however, often has noise or hidden variables which makes approximation inaccurate and in other cases over trained. At best, the prediction settles on the mean of the immediate vicinity. In **Fig. 1** we can see that using a neural network to approximate a noisy data set fails to capture all the information. Despite being able to predict a value for any input on the dataset that will be the closest to all the possibilities, the actual distribution of the outputs which may be quite complex, remains a mystery.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig1.png" alt="fig1" style="zoom:80%;" /> |
| **Fig. 1** Approximation of <img src="https://render.githubusercontent.com/render/math?math=ax%5E3%20%2B%20bx%5E2%20%2B%20cx%20%2B%20d"> with added normal noise |




Because of this it is useful to a have a model that instead of producing a single prediction value for a given input value, produces an arbitrary set of predictions mirroring the dataset distribution at that input.

## The Method

Given that the function producing the data has a specific distribution <img src="https://render.githubusercontent.com/render/math?math=Y"> at input <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X"> [^1] we can define the target function, or the function we actually want to approximate, as <img src="https://render.githubusercontent.com/render/math?math=y%20%5Csim%20Y_%7Bx%7D">. We want to create an algorithm capable of sampling an arbitrary number of data points from <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> [^2] for any given <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X">.

To do this we introduce a secondary input <img src="https://render.githubusercontent.com/render/math?math=z"> that can be sampled from a uniformly distributed space <img src="https://render.githubusercontent.com/render/math?math=Z"> [^3] by the algorithm and fed to a deterministic function <img src="https://render.githubusercontent.com/render/math?math=f%28x%2C%20z%29%20%3D%20y"> where <img src="https://render.githubusercontent.com/render/math?math=y"> matches an observation of the real data with probability <img src="https://render.githubusercontent.com/render/math?math=p_%7BZ%7D%28z%29">.

Or put another way, we want a deterministic function that for any given input <img src="https://render.githubusercontent.com/render/math?math=x">, maps a random (but uniform) variable <img src="https://render.githubusercontent.com/render/math?math=Z"> to a dependent random variable <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D">.

[^1]: <img src="https://render.githubusercontent.com/render/math?math=X"> is the n-dimensional continuous domain of the target function.  
[^2]: <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> is a dependent random variable in an n-dimensional continuous space. The probability function must be continuous on <img src="https://render.githubusercontent.com/render/math?math=x">. In this article and the provided source code <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> is assumed to be 1-dimensional.  
[^3]: <img src="https://render.githubusercontent.com/render/math?math=Z"> is a uniformly distributed random variable in an n-dimensional continuous space with a predefined range, however in this article and the provided code <img src="https://render.githubusercontent.com/render/math?math=Z"> is always assumed to be 1-dimensional.


## Model

The proposed model to approximate <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> is an ordinary feed forward neural network that in addition to an input <img src="https://render.githubusercontent.com/render/math?math=x"> takes an input <img src="https://render.githubusercontent.com/render/math?math=z"> that can be sampled from <img src="https://render.githubusercontent.com/render/math?math=Z">.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\model.png" alt="model" /> |
### Overview

At every point <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X"> we want <img src="https://render.githubusercontent.com/render/math?math=f%28x%2C%20z%20%5Csim%20Z%29"> to approximate the <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> distribution to an arbitrary precision. Let's picture <img src="https://render.githubusercontent.com/render/math?math=f%28x%2C%20z%20%5Csim%20Z%29"> and <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> as 2-dimensional (in the case where <img src="https://render.githubusercontent.com/render/math?math=X"> and <img src="https://render.githubusercontent.com/render/math?math=Z"> are both 1-dimensional) pieces of fabric, they can stretch and shrink in different measures at different regions, decreasing or increasing it's density respectively. We want a mechanism that stretches and shrinks <img src="https://render.githubusercontent.com/render/math?math=f%28x%2C%20z%20%5Csim%20Z%29"> in a way that matches the shrinks and stretches in <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D">.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig_x2_uniform.gif" alt="const_uniform" style="zoom:66%;" /> |
| **Fig 2** An animation of the training to match <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> with added uniform noise. |

In **Fig 2** we can see how the trained model output stretches and shrinks little by little on each epoch until it matches target function.

Going on with the stretching and shrinking piece of fabric analogy, we want to put "pins" into an overlaying fabric so that we can superimpose it over the underlying fabric that we are trying to match. We will put these pins into fixed points in the overlaying fabric but we will move them to different places of the underlying fabric as we train the model. At first we will pin them to random places on the underlying fabric. As we observe the position of the pins on the underlying fabric relative to the overlaying fabric we will move them slightly upwards or downwards to improve the overlaying fabric's match on the underlying fabric. Every pin will affect its surroundings in the fabric proportionally to distance from the pin.

We'll start by putting 1 pin at a fixed position in any given longitude of the overlaying fabric and at the midpoint latitude across the fabric's height. We'll then make many observations in the underlying fabric at the same longitude, that is we will randomly pick several locations at the vertical line that goes through the selected pin location.

For every observed point, we'll move the pin position on the underlying fabric (keeping the same fixed position on the overlaying fabric) a small predefined distance downwards if the observed point is below its current position, and we'll move it upwards if it is above it. This means that if there are more observed points above the pin's position in the underlying fabric the total movement will be upwards and vice versa if there are more observed points below it. If we repeat this process enough times, the pin's position in the underlying fabric will settle in a place that divides the observed points by half, that is the same amount of observed points are above it as below it.

> ***Why do we move the pin a predefined distance up or down instead of a distance proportional to the observed point?***  *The reason is that we are not interested in matching the observed point. Since the target dataset is stochastic, matching a random observation is pointless. The information we get from the observed points is whether or not the pin divides them equally (or by another specific ratio)*

|                                                              |
| :----------------------------------------------------------: |
|  <img src="images\fig3.gif" alt="fig3" style="zoom:50%;" />  |
| **Fig 3** Moving 1 pin towards observed points until it settles. |

The pin comes to a stable position dividing all data points in half because the amount of movement for every observation is equal for data points above and data points below. If the predefined distance of movement for observations above is different from the predefined distance of movement for observations below then the pin would settle in a position dividing the data points by a different ratio (different than half). For example, let's try having 2 pins instead of 1, the first one will move 1 distance for observations above it and 0.5 distance for observations below, the second one will do the opposite. After enough iterations the first one should settle at a position that divides the data points by <img src="https://render.githubusercontent.com/render/math?math=1/3"> above and <img src="https://render.githubusercontent.com/render/math?math=2/3"> below while the second pin will divide by <img src="https://render.githubusercontent.com/render/math?math=2/3"> above and <img src="https://render.githubusercontent.com/render/math?math=1/3"> below. This means we'll have <img src="https://render.githubusercontent.com/render/math?math=1/3"> above the first pin, <img src="https://render.githubusercontent.com/render/math?math=1/3"> between both pins and <img src="https://render.githubusercontent.com/render/math?math=1/3"> below the second pin.

|                                                              |
| :-: |
| <img src="images\fig4.gif" alt="fig4" style="zoom:50%;" /> |
| **Fig 4** Moving 2 pins towards observed points until they settle. |

If  a pin divides the observed data points in 2 groups of sizes <img src="https://render.githubusercontent.com/render/math?math=a"> and <img src="https://render.githubusercontent.com/render/math?math=b"> and after training its fixed position settles in the underlying fabric in the <img src="https://render.githubusercontent.com/render/math?math=a/%28a%2Bb%29"> latitude from the top, we have a single point mapping between the 2 fabrics, that is at this longitude the densities above and below the pin are equal in both pieces of fabric. We can extrapolate this concept and create as many pins as we want in order to create a finer mapping between the 2 pieces of fabric.

### Definitions

We'll start by selecting a fixed set of points in <img src="https://render.githubusercontent.com/render/math?math=Z">  of size <img src="https://render.githubusercontent.com/render/math?math=S"> that we will call ***z-samples***. We can define this set as:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D%20%3D%20%5C%7Bz_0%2C%20z_%7B1%7D%2C%20...%20%2C%20z_%7BS%7D%5C%7D%20%5Cin%20Z%20%5C%20s.t.%20%5C%20z_%7B0%7D%20%3C%20z_%7B1%7D%20%3C%20%5C%20...%20%5C%20%3C%20z_%7BS%7D%0A">|

The predictive model will be defined as:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%2C%20z%20%5Csim%20Z%3A%20f_%7B%5Ctheta%7D%28x%2C%20z%29%0A">|

Here <img src="https://render.githubusercontent.com/render/math?math=x"> will be any input tuple from the input domain, <img src="https://render.githubusercontent.com/render/math?math=z"> will be a sample from uniform random variable <img src="https://render.githubusercontent.com/render/math?math=Z"> and <img src="https://render.githubusercontent.com/render/math?math=%5Ctheta"> is the internal state of the model, or the weight matrix.

Then we define the prediction error for any input <img src="https://render.githubusercontent.com/render/math?math=x"> for a specific <img src="https://render.githubusercontent.com/render/math?math=z%20%5Cin%20Z"> as:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20z%20%5Cin%20Z%3A%20E_%7B%5Ctheta%7D%28z%29%20%3D%20P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%29%29%20-%20P%28Z%20%3C%3D%20z%29%20%5Ctag%7B1%7D%0A">|

That is, the difference between the real data cumulative probability distribution and the predicted cumulative probability distribution.

Now we can define our training goals as:

###### Goal 1

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z%27%20%5Cin%20z_%7Bsamples%7D%3A%20arg%20%5Cmin_%7B%5Ctheta%7D%20%7CE_%7B%5Ctheta%7D%28z%27%29%7C%20%5Ctag%7B2%7D%0A">|

In other words, we want that for every <img src="https://render.githubusercontent.com/render/math?math=z%27"> in <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> and across the entire <img src="https://render.githubusercontent.com/render/math?math=X"> input space the absolute error <img src="https://render.githubusercontent.com/render/math?math=%7CE_%7B%5Ctheta%7D%28z%27%29%7C"> is minimized. This first goal gives us an approximate discrete finite mapping between the ***z-samples*** set and <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D">. Even if it doesn't say anything about all the points <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7Bz%7D"> in <img src="https://render.githubusercontent.com/render/math?math=Z"> that are not in <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D">.

###### Goal 2

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z_%7B0%7D%2C%20z_%7B1%7D%20%5Cin%20Z%20%5Cspace%20s.t.%20%5Cspace%20z_%7B0%7D%20%3C%20z_%7B1%7D%3A%20f_%7B%5Ctheta%7D%28x%2C%20z_%7B0%7D%29%20%3C%20f_%7B%5Ctheta%7D%28x%2C%20z_%7B1%7D%29%20%5Ctag%7B3%7D%0A">|

This second goal gives us that for any given <img src="https://render.githubusercontent.com/render/math?math=x"> in <img src="https://render.githubusercontent.com/render/math?math=X">, <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> is a monotonically increasing function in <img src="https://render.githubusercontent.com/render/math?math=Z">.

Both of these goals will be tested empirically during the testing step of the training algorithm.

For any point <img src="https://render.githubusercontent.com/render/math?math=z%20%5Csim%20Z"> and with <img src="https://render.githubusercontent.com/render/math?math=z%27"> and <img src="https://render.githubusercontent.com/render/math?math=z%27%27"> being the ***z-samples*** that are immediately smaller and greater respectively, and assuming [Goal 2](#Goal-2)  is met we have:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z%20%5C%20st%3A%20%5C%20z%27%20%3C%20z%20%3C%20z%27%27%3A%20P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%27%29%29%20%3C%20%5Cmathbf%7BP%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%29%29%7D%20%3C%20P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%27%27%29%29%20%5Ctag%7B5%7D%0A">|

and replacing the prediction error we have:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z%20%5C%20st%3A%20%5C%20z%27%20%3C%20z%20%3C%20z%27%27%3A%20P%28Z%20%3C%3D%20z%27%29%20%2B%20E_%7B%5Ctheta%7D%28z%27%29%20%3C%20%5Cmathbf%7BP%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%29%29%7D%20%3C%20P%28Z%20%3C%3D%20z%27%27%29%20%2B%20E_%7B%5Ctheta%7D%28z%27%27%29%20%5Ctag%7B6%7D%0A">|

And if we substract <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%29"> from every term we have:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z%20%5C%20st%3A%20%5C%20z%27%20%3C%20z%20%3C%20z%27%27%3A%20P%28Z%20%3C%3D%20z%27%29%20-%20P%28Z%20%3C%3D%20z%29%20%2B%20E_%7B%5Ctheta%7D%28z%27%29%20%3C%20%5Cmathbf%7BE_%7B%5Ctheta%7D%28z%29%7D%20%3C%20P%28Z%20%3C%3D%20z%27%27%29%20-%20P%28Z%20%3C%3D%20z%29%20%2B%20E_%7B%5Ctheta%7D%28z%27%27%29%20%5Ctag%7B7%7D%0A">|

What this means is that for any point <img src="https://render.githubusercontent.com/render/math?math=z%20%5Csim%20Z"> the prediction error error <img src="https://render.githubusercontent.com/render/math?math=E_%7B%5Ctheta%7D%28z%29"> is lower bounded by <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29%20-%20P%28Z%20%3C%3D%20z%29%20%2B%20E_%7B%5Ctheta%7D%28z%27%29"> and upper bounded by <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%27%29%20-%20P%28Z%20%3C%3D%20z%29%20%2B%20E_%7B%5Ctheta%7D%28z%27%27%29">.

Assuming [Goal 1](#Goal-1) is met we know that <img src="https://render.githubusercontent.com/render/math?math=E_%7B%5Ctheta%7D%28z%27%29"> and <img src="https://render.githubusercontent.com/render/math?math=E_%7B%5Ctheta%7D%28z%27%27%29"> are minimized which leaves <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29%20-%20P%28Z%20%3C%3D%20z%29"> and <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%27%29%20-%20P%28Z%20%3C%3D%20z%29"> as the dominant factors. The distance between any <img src="https://render.githubusercontent.com/render/math?math=z"> and its neighboring **z-samples** can be minimized by increasing the number of ***z-samples*** or <img src="https://render.githubusercontent.com/render/math?math=S">. In other words the maximum error of <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> can be arbitrarily minimized by a sufficiently large <img src="https://render.githubusercontent.com/render/math?math=S">.


Having defined our goals and what will they buy us, we move to show how we will achieve [Goal 1](#Goal-1). For simplicity we will use a ***z-samples*** set that is evenly distributed in <img src="https://render.githubusercontent.com/render/math?math=Z">, that is: <img src="https://render.githubusercontent.com/render/math?math=%5C%7Bz_0%2C%20z_%7B1%7D%2C%20...%20%2C%20z_%7BS%20-%201%7D%5C%7D%20%5Cin%20Z%20s.t.%20z_%7B0%7D%20%3C%20z_%7B1%7D%2C%20...%20%2C%20%3C%20z_%7BS%7D%20%5Cand%20P%28z_%7B0%7D%20%3C%20Z%20%3C%20z_%7B1%7D%29%20%3D%20P%28z_%7B1%7D%20%3C%20Z%20%3C%20z_%7B2%7D%29%20%3D%20%5C%20...%20%5C%20%3D%20P%28z_%7Bs-1%7D%20%3C%20Z%20%3C%20z_%7Bs%7D%29">.

For any given <img src="https://render.githubusercontent.com/render/math?math=x"> in <img src="https://render.githubusercontent.com/render/math?math=X"> and any <img src="https://render.githubusercontent.com/render/math?math=z%27"> in <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> we want <img src="https://render.githubusercontent.com/render/math?math=f"> to satisfy <img src="https://render.githubusercontent.com/render/math?math=P%28Y_%7Bx%7D%20%3C%3D%20f%28x%2C%20z%27%29%29%20%3D%20P%28Z%20%3C%3D%20z%27%29">. For this purpose we'll assume that we count with a sufficiently representative set of samples in <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> or <img src="https://render.githubusercontent.com/render/math?math=y_%7Btrain%7D%20%5Csim%20Y_%7Bx%7D">.

For a given <img src="https://render.githubusercontent.com/render/math?math=%5Cbar%7Bx%7D%20%5Cin%20X"> if <img src="https://render.githubusercontent.com/render/math?math=z%27"> was the midpoint in <img src="https://render.githubusercontent.com/render/math?math=Z"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=z%27%20%5Cin%20Z%20s.t.%20Pr%28z%27%20%3E%20Z%29%20%3D%200.5">) we could simply train <img src="https://render.githubusercontent.com/render/math?math=f"> to change the value of <img src="https://render.githubusercontent.com/render/math?math=f%28%5Cbar%7Bx%7D%2C%20z%27%29"> a constant movement number <img src="https://render.githubusercontent.com/render/math?math=M"> greater for every training example <img src="https://render.githubusercontent.com/render/math?math=y%20%5Cin%20y_%7Btrain%7D"> that was greater than <img src="https://render.githubusercontent.com/render/math?math=f%28%5Cbar%7Bx%7D%2C%20z%27%29"> itself and the same constant number smaller for every training example that was smaller. This would cause after enough iterations for the value of <img src="https://render.githubusercontent.com/render/math?math=f%28%5Cbar%7Bx%7D%2Cz%27%29"> to settle in a position that divides in half all training examples when the total movement equals 0.

If instead of being <img src="https://render.githubusercontent.com/render/math?math=Z">'s midpoint <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29%20%5Cne%200.5"> then the constant numbers of movement for greater and smaller samples have to be different.

Let's say that <img src="https://render.githubusercontent.com/render/math?math=a"> is the distance between <img src="https://render.githubusercontent.com/render/math?math=z%27"> and the smallest number in <img src="https://render.githubusercontent.com/render/math?math=Z"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D">, and <img src="https://render.githubusercontent.com/render/math?math=b"> the distance between <img src="https://render.githubusercontent.com/render/math?math=z%27"> and <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D">.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Barray%7D%7Bll%7D%0Aa%20%3D%20z%27%20-%20Z_%7Bmin%7D%5C%5C%0Ab%20%3D%20Z_%7Bmax%7D%20-%20z%27%0A%5Cend%7Barray%7D%0A">|



Since <img src="https://render.githubusercontent.com/render/math?math=a"> represents the amount of training examples we hope to find smaller than <img src="https://render.githubusercontent.com/render/math?math=z%27"> and <img src="https://render.githubusercontent.com/render/math?math=b"> the amount of training examples greater than <img src="https://render.githubusercontent.com/render/math?math=z%27"> we need 2 scalars <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> to satisfy the following equations:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20a%20%3D%20%5Cbeta%20b%20%5Ctag%7B8%7D%0A">|

These scalars will be the multipliers to be used with the constant movement <img src="https://render.githubusercontent.com/render/math?math=M"> on every observed point smaller and greater than <img src="https://render.githubusercontent.com/render/math?math=z%27"> respectively. This first equation assures that the total movement when <img src="https://render.githubusercontent.com/render/math?math=z%27"> is situated at <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D%20%2B%20a"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D%20-%20b"> will be 0.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20a%20%2B%20%5Cbeta%20b%20%3D%201%20%5Ctag%7B9%7D%0A">|

This second equation normalizes the scalars so that the total movement for all <img src="https://render.githubusercontent.com/render/math?math=z"> in <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> have the same total movement.

Which gives us:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Barray%7D%7Bll%7D%0A%5Calpha%20%3D%201%20/%20%282%20%20a%29%5C%5C%0A%5Cbeta%20%3D%201%20/%20%282%20b%29%0A%5Cend%7Barray%7D%20%5Ctag%7B10%7D%0A">|

This logic however, breaks at the edges, that is when a *z-sample* is equal to <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D">. At these values either <img src="https://render.githubusercontent.com/render/math?math=a"> or <img src="https://render.githubusercontent.com/render/math?math=b"> is 0 and if either of them is 0 then one of <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> or <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> is undefined.

As <img src="https://render.githubusercontent.com/render/math?math=a"> or <img src="https://render.githubusercontent.com/render/math?math=b"> approach 0 <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> or <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> tend to infinity, one might be tempted to replace this with a large number, but that would not be practical because a large distance multiplier would dominate the training and minimize the movement of other <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D">.

Also as one of <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> or <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> tend to infinity the other one becomes a small number that is also impractical but for a different reason. The <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> at the edges are supposed to map to the edges of <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> and any quantity of movement into the opposite direction will result in <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D"> mapping to a greater or smaller point in <img src="https://render.githubusercontent.com/render/math?math=Y_x"> respectively. For this reason the <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> for the <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> at the edges (i.e. <img src="https://render.githubusercontent.com/render/math?math=z_%7B0%7D"> and <img src="https://render.githubusercontent.com/render/math?math=z_%7BS%7D">) will be assigned a value of 0 for the one that pushes inward and a predefined constant <img src="https://render.githubusercontent.com/render/math?math=C%20%5Cin%20%5B0%2C%201%5D"> that can be adjusted to the model.

## Training the model

In order to train the neural network the *z-samples* set size <img src="https://render.githubusercontent.com/render/math?math=S"> is chosen depending on the desired accuracy and compute available. Having decided the training level <img src="https://render.githubusercontent.com/render/math?math=Z"> must be defined. That is, the number of dimensions and its range must be chosen. Given <img src="https://render.githubusercontent.com/render/math?math=Z"> and the training level we can create the **z-samples** set.

For example if <img src="https://render.githubusercontent.com/render/math?math=Z"> is 1-dimensional with range defined as <img src="https://render.githubusercontent.com/render/math?math=%5B10.0%2C%2020.0%5D"> and <img src="https://render.githubusercontent.com/render/math?math=S%20%3D%209">, we have that the *z-sample* set is <img src="https://render.githubusercontent.com/render/math?math=%5C%7Bz_%7B0%7D%20%2810.0%29%2C%20z_%7B1%7D%20%2811.25%29%2C%20z_%7B2%7D%20%2812.5%29%2C%20z_%7B3%7D%20%2813.75%29%2C%20z_%7B4%7D%20%2815.0%29%2C%20z_%7B5%7D%20%2816.25%29%2C%20z_%7B6%7D%20%2817.5%29%2C%20z_%7B7%7D%20%2818.75%29%2C%20z_%7B8%7D%20%2820.0%29%5C%7D">.

First we select a batch of data from the training data with size <img src="https://render.githubusercontent.com/render/math?math=n">, for every data point in the batch, we evaluate the current model on every *z-sample*. This gives us the prediction matrix:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Af_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7BS%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7BS%7D%29%5C%5C%0A...%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bn%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bn%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7Bn%7D%2C%20z_%7BS%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

For every data point <img src="https://render.githubusercontent.com/render/math?math=%28x_%7Bi%7D%2C%20y_%7Bi%7D%29"> in the batch, we take the output value <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> and compare it with every value of its corresponding row in the prediction matrix (i.e. <img src="https://render.githubusercontent.com/render/math?math=%5Bf_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7B0%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7B8%7D%29%5D">). After determining if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is greater or smaller than each predicted value, we produce 2 values for every element in the matrix:

###### Scalar
The scalar will be <img src="https://render.githubusercontent.com/render/math?math=%5Calpha_%7Bz-sample%7D"> if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is smaller than the prediction and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta_%7Bz-sample%7D"> if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is greater.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=s_%7Bi%2C%20z-sample%7D%20%3D%20%5Cleft%5C%7B%0A%5Cbegin%7Barray%7D%7Bll%7D%0A%5Calpha_%7Bz-sample%7D%20%26%20y_%7Bi%7D%20%3C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%5C%5C%0A%5Cbeta_%7Bz-sample%7D%20%26%20y_%7Bi%7D%20%3E%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%5C%5C%0A%5Cend%7Barray%7D%0A%5Cright.%0A">|

###### Target Value
The target value is the prediction itself plus the preselected movement constant <img src="https://render.githubusercontent.com/render/math?math=M"> multiplied by -1 if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is smaller than the prediction and 1 if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is greater. You can think of target values as the "where we want the prediction to be" value.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=t_%7Bi%2C%20z-sample%7D%20%3D%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%0Af_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%20%2B%20M%20%26%20y_%7Bi%7D%20%3C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%20-%20M%20%26%20y_%7Bi%7D%20%3E%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%5C%5C%0A%5Cend%7Barray%7D%5Cright.%0A">|

After calculating these 2 values we are ready to assemble the matrix to be used during backpropagation.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0A%28w_%7B0%2C0%7D%2C%20t_%7B0%2C0%7D%29%2C%20%28w_%7B0%2C1%7D%2C%20t_%7B0%2C1%7D%29%2C%20...%20%2C%20%28w_%7B0%2CS%7D%2C%20t_%7B0%2CS%7D%29%5C%5C%0A%28w_%7B1%2C0%7D%2C%20t_%7B1%2C0%7D%29%2C%20%28w_%7B1%2C1%7D%2C%20t_%7B1%2C1%7D%29%2C%20...%20%2C%20%28w_%7B1%2CS%7D%2C%20t_%7B1%2CS%7D%29%5C%5C%0A...%5C%5C%0A%28w_%7Bn%2C0%7D%2C%20t_%7Bn%2C0%7D%29%2C%20%28w_%7Bn%2C1%7D%2C%20t_%7Bn%2C1%7D%29%2C%20...%20%2C%20%28w_%7Bn%2CS%7D%2C%20t_%7Bn%2CS%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

We pass the prediction matrix results in a addition to this matrix to a Weighted Mean Squared Error loss function (WMSE). The loss function will look like this:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Csum_%7Bj%3D0%7D%5E%7BS%7D%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%28f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bj%7D%29%20-%20t_%7Bi%2Cj%7D%29%5E2%20%2A%20s_%7Bi%2Cj%7D%0A">|



## Testing the model

The Mean Squared Error (MSE) loss function works to train the model using backpropagation and target values, but testing the model requires a different approach. Since both <img src="https://render.githubusercontent.com/render/math?math=Z"> and <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> are random variables, measuring the differences between them is pointless. Because of this, the success of the model will be measured in 2 ways:

### Earth Movers Distance (EMD)

> In statistics, the **earth mover's distance** (**EMD**) is a measure of the distance between two probability distributions over a region *D*. In mathematics, this is known as the Wasserstein metric. Informally, if the distributions are interpreted as two different ways of piling up a certain amount of dirt over the region *D*, the EMD is the minimum cost of turning one pile into the other; where the cost is assumed to be amount of dirt moved times the distance by which it is moved. [wikipedia.org][EMD]

Using EMD we can obtain an indicator of how similar <img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%3A%20y%20%5Csim%20Y_%7Bx%7D"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%3A%20f_%7B%5Ctheta%7D%28x%2C%20z%20%5Csim%20Z%29"> are. It can be calculated by comparing every <img src="https://render.githubusercontent.com/render/math?math=%28x%2C%20y%29"> data point in the test data and prediction data sets and finding way to transform one into the other that requires the smallest total movement. What the EMD number tells us is the average amount of distance to transform every point in the predictions data set to the test data set.

On the example below you can see that the mean EMD is ~3.9 on a data set with a thickness of roughly 100. Because of the random nature of the data sets the EMD cannot be used as a literal error indicator, but it can be used as a progress indicator, that is to tell if the model improves with training.

|      |
| :-: |
|   <img src="images\fig5.png" alt="fig5" style="zoom:50%;" />   |
| **Fig 5** EMD testing. |

### Testing the training goals
#### Training goal 1

Ideally to test [Goal 1](#Goal-1) (i.e. <img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z%27%20%5Cin%20z_%7Bsamples%7D%3A%20arg%20%5Cmin_%7B%5Ctheta%7D%20%7CE_%7B%5Ctheta%7D%28z%27%29%7C">) we would evaluate <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> for a given <img src="https://render.githubusercontent.com/render/math?math=x"> and on every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> and then compare it to an arbitrary number of test data points having the same <img src="https://render.githubusercontent.com/render/math?math=x">. We would then proceed to count for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> the number of test data points smaller than it. With a vector of *smaller than counts* (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%27%29%29">) we could proceed to compare it with the canonical counts for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29">) and measure the error. In real life data sets this is not possible however. Real life data sets will not likely have an arbitrary number of data points having the same <img src="https://render.githubusercontent.com/render/math?math=x"> (they will unlikely have 2 data points with the same <img src="https://render.githubusercontent.com/render/math?math=x">) which means that we need to use a vicinity in <img src="https://render.githubusercontent.com/render/math?math=x"> (values <img src="https://render.githubusercontent.com/render/math?math=X"> that are close to an <img src="https://render.githubusercontent.com/render/math?math=x">) to test the goal.

We start by creating an ordering (an array of indices) <img src="https://render.githubusercontent.com/render/math?math=O_%7Bd%7D%20%3D%20%5C%7Bo_%7B0%7D%2C%20o_%7B1%7D%2C%20...%2C%20o_%7Bm%7D%5C%7D"> that sorts all the elements in <img src="https://render.githubusercontent.com/render/math?math=X_%7Btest%7D"> (all the <img src="https://render.githubusercontent.com/render/math?math=x"> inputs in the test data set) on dimension <img src="https://render.githubusercontent.com/render/math?math=d">. Then we select an element in <img src="https://render.githubusercontent.com/render/math?math=O_%7Bd%7D"> and pick the <img src="https://render.githubusercontent.com/render/math?math=V"> (vicinity size) samples closest to it. This gives us a subset of  consecutive elements in <img src="https://render.githubusercontent.com/render/math?math=O_%7Bd%7D"> which we'll call <img src="https://render.githubusercontent.com/render/math?math=G%20%3D%20%5C%7Bo_%7B0%7D%2C%20o_%7B1%7D%2C%20...%20%2C%20o_%7BV%7D%5C%7D">.

Now we can evaluate <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> for every <img src="https://render.githubusercontent.com/render/math?math=x_%7Bo%27%7D%20%5Cmid%20o%27%20%5Cin%20G"> on every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> which gives us the matrix:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Af_%7B%5Ctheta%7D%28x_%7Bo_%7B0%7D%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7B0%7D%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7B0%7D%7D%2C%20z_%7BS%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bo_%7B1%7D%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7B1%7D%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7B1%7D%7D%2C%20z_%7BS%7D%29%5C%5C%0A...%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bo_%7BV%7D%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7BV%7D%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7BV%7D%7D%2C%20z_%7BS%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

We then proceed to compare each row with the outputs <img src="https://render.githubusercontent.com/render/math?math=y_%7Bo%27%7D%20%5Cmid%20o%27%20%5Cin%20G">

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Ay_%7Bo_%7B0%7D%7D%5C%5C%0Ay_%7Bo_%7B1%7D%7D%5C%5C%0A...%5C%5C%0Ay_%7Bo_%7BV%7D%7D%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

and create *smaller than counts* (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%29%29">) which we can then compare with the canonical counts for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%29">) to measure the error in the selected vicinity.

We will create a number of such vicinities and call each error as the local errors and a vicinity covering all <img src="https://render.githubusercontent.com/render/math?math=X_%7Btest%7D"> and call its error the mean error.

On the example below you can see that the goal 1 error is ~1.6%, this can be used as an error indicator for the model.

|      |
| :-: |
|   <img src="images\fig6.png" alt="fig6" style="zoom:50%;" />   |
| **Fig 6** Training goal 1 testing. |



#### Training goal 2

In order to test [Goal 2](#Goal-2) <img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cand%20%5Cforall%20z_%7B0%7D%2C%20z_%7B1%7D%20%5Cin%20Z%20%5Cspace%20s.t.%20%5Cspace%20z_%7B0%7D%20%3C%20z_%7B1%7D%3A%20f_%7B%5Ctheta%7D%28x%2C%20z_%7B0%7D%29%20%3C%20f_%7B%5Ctheta%7D%28x%2C%20z_%7B1%7D%29"> we select some random points in <img src="https://render.githubusercontent.com/render/math?math=X"> and a set of random points in <img src="https://render.githubusercontent.com/render/math?math=Z">, we run them in our model and get result matrix:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Af_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7Bn%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7Bn%7D%29%5C%5C%0A...%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bm%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bm%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bm%7D%2C%20z_%7Bn%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|


From here it is trivial to check that each row is monotonically increasing. To increase the quality of the check we can increment the sizes of the test point set in <img src="https://render.githubusercontent.com/render/math?math=X"> and the test point set in <img src="https://render.githubusercontent.com/render/math?math=Z">.

## Experiments

The following are various experiments done on different datasets.

### <img src="https://render.githubusercontent.com/render/math?math=%7B%5Clarge%20x%5E2%7D"> plus gaussian noise

Let's start with a simple example. The function <img src="https://render.githubusercontent.com/render/math?math=x_%7B2%7D"> with added gaussian noise. On the left panel you can see the training evolving over the course of 180 epochs. On the top left corner of this panel you can see the goal 1 error localized over <img src="https://render.githubusercontent.com/render/math?math=X">, at the end of the training you can see that the highest local error is around 2% and the global error is around 0.5%. On the top right corner of the same panel you can see the local Earth Mover's Distance (EMD). On the bottom left corner you can see a plot of the original test dataset (in blue) and the <img src="https://render.githubusercontent.com/render/math?math=z-lines"> (in orange), you can see how they progressively conform to the test data. On the bottom right you can see a plot of the original test dataset (in blue) and random predictions (with <img src="https://render.githubusercontent.com/render/math?math=z%20%5Csim%20Z">), you can see as the predicted results progressively represent the test data.

On the right panel, you can see a plot of the global goal1 error (above) and global EMD values (below) as they change over the course of the training.

|      |      |
| ---- | ---- |
|   <img src="images\x2_normal_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_x2norm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 7** Training model to match <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> plus gaussian. |  |

### <img src="https://render.githubusercontent.com/render/math?math=%7B%5Clarge%20a%20x%5E3%20%2B%20bx%5E2%20%2B%20cx%20%2B%20d%7D"> plus truncated gaussian noise

This one is a bit more complicated. An order 3 polynomial with added truncated gaussian noise, that is a normal distribution clipped at specific points.

|      |      |
| ---- | ---- |
|   <img src="images\x3x2_trunc_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_x3x2trunc_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 8** Training model to match <img src="https://render.githubusercontent.com/render/math?math=x%5E3%20%2B%20bx%5E2%20%2B%20cx%20%2B%20d"> plus truncated gaussian. |  |

### Double <img src="https://render.githubusercontent.com/render/math?math=%7B%5Clarge%20sin%28x%29%7D"> plus gaussian noise multiplied by sin(x)

This one is quite more interesting. 2 mirroring <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29"> functions with gaussian noise scaled by <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29"> itself.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=f%20%3D%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%0Asin%28x%29%20%2B%20%5Cmathcal%7BN%7D%20%2A%20sin%28x%29%20%26%20U%280%2C1%29%20%3C%3D%200.5%5C%5C%0A-sin%28x%29%20%2B%20%5Cmathcal%7BN%7D%20%2A%20sin%28x%29%20%26%20U%280%2C1%29%20%3E%200.5%5C%5C%0A%5Cend%7Barray%7D%5Cright.%0A">|



Notice how the model succeeds to represent the areas in the middle with lower densities.


|      |      |
| ---- | ---- |
|   <img src="images\sin_sin_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_sinsin_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 9** Training model to match double <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29"> with added gaussian times <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29">. |  |

### Branching function plus gaussian noise

This one experiments with branching paths. It starts with simple gaussian noise around <img src="https://render.githubusercontent.com/render/math?math=0">, then starts splitting it with equal probabilities over the course of various segments.


| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=f%20%3D%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%0A0.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B-4%2C%20-2%5D%5C%5C%0A1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B-2%2C%200%5D%20%26%5Cand%20%26U%280%2C1%29%20%3C%3D%200.5%5C%5C%0A-1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B-2%2C%200%5D%20%26%5Cand%20%260.5%20%3C%20U%280%2C1%29%5C%5C%0A0.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B0%2C%202%5D%20%26%5Cand%20%260.5%20%3C%20U%280%2C1%29%5C%5C%0A2.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B0%2C%202%5D%20%26%5Cand%20%260.25%20%3C%20U%280%2C1%29%20%3C%3D%200.5%5C%5C%0A-2.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B0%2C%202%5D%20%26%5Cand%20%26U%280%2C1%29%20%3C%3D%200.25%5C%5C%0A1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%26U%280%2C1%29%20%3C%3D%200.375%5C%5C%0A-1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%260.375%20%3C%20U%280%2C1%29%20%3C%3D%200.75%5C%5C%0A3.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%260.75%20%3C%20U%280%2C1%29%20%3C%3D%200.875%5C%5C%0A-3.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%260.875%20%3C%20U%280%2C1%29%5C%5C%0A%5Cend%7Barray%7D%5Cright.%0A">|


Despite the distribution not being continuous, the model does a reasonably good job of approximating it.

|      |      |
| ---- | ---- |
|   <img src="images\branch_norm_plots_res.gif" alt="fig7" style="zoom:80%;" />   |   <img src="images\fig_branchnorm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 10** Training model to match branching function plus gaussian. |  |

### <img src="https://render.githubusercontent.com/render/math?math=%7B%5Clarge%20x_%7B0%7D%5E2%20%2B%20x_%7B1%7D%5E3%7D"> plus absolute gaussian noise

The next example has 2 dimensions of input. <img src="https://render.githubusercontent.com/render/math?math=X_%7B0%7D"> (the first dimension) is <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> and <img src="https://render.githubusercontent.com/render/math?math=X_%7B1%7D"> (the second dimension) is <img src="https://render.githubusercontent.com/render/math?math=x%5E3"> with added absolute gaussian noise. The display is slightly different, for the sake of space the <img src="https://render.githubusercontent.com/render/math?math=z-lines"> plot is omitted. As you can see there is a panel per dimension and as always an additional panel for goal 1 error and EMD error histories.

|      |
| ---- |
|<img src="images\x3_x2_absnormal_plots_res_0.gif" alt="fig10_0" style="zoom:80%;" />|
|<img src="images\x3_x2_absnormal_plots_res_1.gif" alt="fig10_1" style="zoom:80%;" />|
|<img src="images\fig_x3x2abs_tensorboard.png" alt="fig7" style="zoom:50%;" />|
| **Fig 11** Training model to match <img src="https://render.githubusercontent.com/render/math?math=x_%7B0%7D%5E2%20%2B%20x_%7B1%7D%5E3"> plus absolute gaussian. |

### California housing dataset

This experiment uses real data instead of generated one which proves the model's effectivity on real data. It is the classic California housing dataset. It has information from the 1990 California census with 8 input dimensions (Median Income, House Age, etc ...).

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
| **Fig 12** Training model to match the California housing dataset. |

## Conclusion

The method presented allows to approximate stochastic data sets to an arbitrary precision. The model is simple, fast to train and can be implemented with a vanilla feedforward neural network. Its ability to approximate any distribution across an input space makes it a potentially valuable tool for any task that requires prediction.
