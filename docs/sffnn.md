# A generic approach for training probabilistic machine learning models

###### Nicolas Arroyo nicolas.arroyo.duran@gmail.com



## Introduction

Neural networks are [universal function approximators][UAT]. Which means that having enough hidden neurons a neural network can be used to approximate any continuous function. Real world data, however, often has noise that in some cases makes producing a single deterministic value prediction insufficient. Take for example the dataset in **Fig. 1a**, it shows the relation between the departure delay and the arrival delay for flights out of JFK International Airport over a period of one year (Subset of [2015 Flight Delays and Cancellations][DEL]).

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig1a.png" alt="fig1a" style="zoom:100%;" /> |
| **Fig. 1a** Departure to arrival delays from JFK |



**Fig. 1a** also shows the prediction of a fully trained neural network that does a good job of approximating the mean value and does provide information about the trend of the dataset. However, it does not help answer questions like *given a departure delay, what is the maximum expected arrival in X% of the flights?* or *given a departure delay, what is the probability that the arrival delay will be longer than Y?* or even more interesting, write a model that samples arrival delay values for given departure delays with the same distribution as the real thing.

There are methods to solve this problem, for example, assuming the model in **Fig 1a** estimates the mean, the standard deviation for the entire data set can be calculated and with those parameters you can produce the expected normal distribution. And if the variance is not constant, that is if the variance changes across the input space, you could use [Logistic Regression with Maximum Likelihood Estimation][MLE] which in a nutshell trains a model that predicts the parameters for a specific distribution function (for example the mean and the standard deviation of the normal or gaussian distribution) at a given input. The problem is that it relies on beforehand knowledge of the dataset distribution, which might be difficult in some cases or too irregular to match a known distribution.

In the case of the departure to arrival delays dataset, we can observe from the plot that the distribution appears to be similar to the normal distribution, so it makes sense to build a Maximum Likelihood Estimation model trying to calculate the parameters of a normal distribution. **Fig 1b** shows a plot of a fully trained model showing the mean,  the mean plus/minus the standard deviation and the mean plus/minus twice the standard deviation. The error of this model is of 2.48% which is quite good. However, the dataset's distribution is not perfectly normal, you can see that the upper tail is slightly longer than the lower tail in addition to other imperfections, which is why the accuracy is not better.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig1b.png" alt="fig1b" style="zoom:100%;" /> |
| **Fig. 1b** Departure to arrival delays from JFK and probabilistic model |

This article presents a generic approach to training probabilistic machine learning models that will produce distributions that adapt to the real data with any distribution it may have, even branching distributions or distributions that change form across the input space.

## The method

Given that the function producing the data has a specific distribution <img src="https://render.githubusercontent.com/render/math?math=Y"> at input <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X"> [^1] we can define the target function, or the function we actually want to approximate, as <img src="https://render.githubusercontent.com/render/math?math=y%20%5Csim%20Y_%7Bx%7D">. We want to create an algorithm capable of sampling an arbitrary number of data points from <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> [^2] for any given <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X">.

To do this we introduce a secondary input <img src="https://render.githubusercontent.com/render/math?math=z"> that can be sampled from a uniformly distributed space <img src="https://render.githubusercontent.com/render/math?math=Z"> [^3] by the algorithm and fed to a deterministic function <img src="https://render.githubusercontent.com/render/math?math=f"> such that <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%20z%29%20%3D%20P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%29%29">.

Or put another way, we want a deterministic function that for any given input <img src="https://render.githubusercontent.com/render/math?math=x">, maps a random (but uniform) variable <img src="https://render.githubusercontent.com/render/math?math=Z"> to a dependent random variable <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D">.

[^1]: <img src="https://render.githubusercontent.com/render/math?math=X"> is the n-dimensional continuous domain of the target function.  
[^2]: <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> is a dependent random variable in an n-dimensional continuous space. The probability function must be continuous on <img src="https://render.githubusercontent.com/render/math?math=x">. The method presented in this article only applies to the case where <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> to be 1-dimensional.  
[^3]: <img src="https://render.githubusercontent.com/render/math?math=Z"> is a uniformly distributed random variable in an n-dimensional continuous space with a predefined range.


## Model

The proposed model to approximate <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> is an ordinary feed forward neural network that in addition to an input <img src="https://render.githubusercontent.com/render/math?math=x"> takes an input <img src="https://render.githubusercontent.com/render/math?math=z"> that can be sampled from <img src="https://render.githubusercontent.com/render/math?math=Z">.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\model.png" alt="model" /> |
### Overview

At every point <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X"> we want our model <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20z%20%5Csim%20Z%29"> to approximate the <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> distribution to an arbitrary precision. Let's picture <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20z%20%5Csim%20Z%29"> and <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> as 2-dimensional (in the case where <img src="https://render.githubusercontent.com/render/math?math=X"> and <img src="https://render.githubusercontent.com/render/math?math=Z"> are both 1-dimensional) pieces of fabric, they can stretch and shrink in different measures at different regions, decreasing or increasing their densities respectively. We want a mechanism that stretches and shrinks <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20z%20%5Csim%20Z%29"> in a way that matches the shrinks and stretches in <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D">.

|                                                              |
| :----------------------------------------------------------: |
| <img src="images\fig_x2_uniform.gif" alt="const_uniform" style="zoom:66%;" /> |
| **Fig 2** An animation of the training to match <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> with added uniform noise. |

In **Fig 2** we can see how the trained model output stretches and shrinks little by little on each epoch until it matches target function.

Going on with the stretching and shrinking piece of fabric analogy, we want to put "pins" into an overlaying fabric (our model) so that we can superimpose it over the underlying fabric (the target data set) that we are trying to match. We will put these pins into fixed points in the overlaying fabric but we will move them to different places of the underlying fabric as we train the model. At first we will pin them to random places on the underlying fabric. As we observe the position of the pins on the underlying fabric relative to the overlaying fabric we will move them slightly upwards or downwards to improve the overlaying fabric's match on the underlying fabric. Every pin will affect its surroundings in the fabric proportionally to distance from the pin.

We'll start by putting 1 pin at a fixed position in any given longitude of the overlaying fabric and at the midpoint latitude across the fabric's height. We'll then make many observations in the underlying fabric at the same longitude, that is we will randomly pick several locations at the vertical line that goes through the selected pin location.

For every observed point, we'll move the pin position on the underlying fabric (keeping the same fixed position on the overlaying fabric) a small predefined distance downwards if the observed point is below its current position, and we'll move it upwards if it is above it. This means that if there are more observed points above the pin's position in the underlying fabric the total movement will be upwards and vice versa if there are more observed points below it. If we repeat this process enough times, the pin's position in the underlying fabric will settle in a place that divides the observed points by half, that is the same amount of observed points are above it as below it.

> ***Why do we move the pin a predefined distance up or down instead of a distance proportional to the observed point?***  *The reason is that we are not interested in matching the observed point. Since the target dataset is stochastic, matching a random observation is pointless. The interesting information we get from the observed points is whether or not the pin divides them by half (or by another specific ratio)*

|                                                              |
| :----------------------------------------------------------: |
|  <img src="images\fig3.gif" alt="fig3" style="zoom:50%;" />  |
| **Fig 3** Moving 1 pin towards observed points until it settles. |

**Fig 3** shows how the pin comes to a stable position dividing all data points in half because the amount of movement for every observation is equal for data points above and data points below. If the predefined distance of movement for observations above is different from the predefined distance of movement for observations below then the pin would settle in a position dividing the data points by a different ratio (different than half). For example, let's try having 2 pins instead of 1, the first one will move 1 distance for observations above it and 0.5 distance for observations below, the second one will do the opposite. After enough iterations the first one should settle at a position that divides the data points by <img src="https://render.githubusercontent.com/render/math?math=1/3"> above and <img src="https://render.githubusercontent.com/render/math?math=2/3"> below while the second pin will divide by <img src="https://render.githubusercontent.com/render/math?math=2/3"> above and <img src="https://render.githubusercontent.com/render/math?math=1/3"> below. This means we'll have <img src="https://render.githubusercontent.com/render/math?math=1/3"> above the first pin, <img src="https://render.githubusercontent.com/render/math?math=1/3"> between both pins and <img src="https://render.githubusercontent.com/render/math?math=1/3"> below the second pin like **Fig 4** shows.

|                                                              |
| :-: |
| <img src="images\fig4.gif" alt="fig4" style="zoom:50%;" /> |
| **Fig 4** Moving 2 pins towards observed points until they settle. |

If a pin divides the observed data points in 2 groups of sizes <img src="https://render.githubusercontent.com/render/math?math=a"> and <img src="https://render.githubusercontent.com/render/math?math=b"> and after training its fixed position settles in the underlying fabric in the <img src="https://render.githubusercontent.com/render/math?math=a/%28a%2Bb%29"> latitude from the top, we have a single point mapping between the 2 fabrics, that is at this longitude the densities above and below the pin are equal in both pieces of fabric. We can extrapolate this concept and use as many pins as we want in order to create a finer mapping between the 2 pieces of fabric.

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

### Model accuracy

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

Assuming [Goal 1](#Goal-1) is met we know that <img src="https://render.githubusercontent.com/render/math?math=E_%7B%5Ctheta%7D%28z%27%29"> and <img src="https://render.githubusercontent.com/render/math?math=E_%7B%5Ctheta%7D%28z%27%27%29"> are small numbers which leaves <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29%20-%20P%28Z%20%3C%3D%20z%29"> and <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%27%29%20-%20P%28Z%20%3C%3D%20z%29"> as the dominant factors. The distance between any <img src="https://render.githubusercontent.com/render/math?math=z"> and its neighboring **z-samples** can be minimized by increasing the number of ***z-samples*** or <img src="https://render.githubusercontent.com/render/math?math=S">. In other words the maximum error of <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> can be arbitrarily minimized by a sufficiently large <img src="https://render.githubusercontent.com/render/math?math=S">.

### Calculating the movement scalars

Having defined our goals and what will they buy us, we move to show how we will achieve [Goal 1](#Goal-1). For simplicity we will use a ***z-samples*** set that is evenly distributed in <img src="https://render.githubusercontent.com/render/math?math=Z">, that is: <img src="https://render.githubusercontent.com/render/math?math=%5C%7Bz%5B0%5D%2C%20z%5B1%5D%2C%20...%20%2C%20z%5BS-1%5D%5C%7D%20%5Cin%20Z%20s.t.%20z%5B0%5D%20%3C%20z%5B1%5D%2C%20...%20%2C%20%3C%20z%5BS%5D%20%5Cwedge%20P%28z%5B0%5D%20%3C%20Z%20%3C%20z%5B1%5D%29%20%3D%20P%28z%5B1%5D%20%3C%20Z%20%3C%20z%5B2%5D%29%20%3D%20%5C%20...%20%5C%20%3D%20P%28z%5Bs-1%5D%20%3C%20Z%20%3C%20z%5Bs%5D%29">.

For any given <img src="https://render.githubusercontent.com/render/math?math=x"> in <img src="https://render.githubusercontent.com/render/math?math=X"> and any <img src="https://render.githubusercontent.com/render/math?math=z%27"> in <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> we want <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> to satisfy <img src="https://render.githubusercontent.com/render/math?math=P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%27%29%29%20%3D%20P%28Z%20%3C%3D%20z%27%29">. For this purpose we'll assume that we count with a sufficiently representative set of samples in <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> or <img src="https://render.githubusercontent.com/render/math?math=y_%7Btrain%7D%20%5Csim%20Y_%7Bx%7D">.

For a given <img src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20X"> and having <img src="https://render.githubusercontent.com/render/math?math=z%27"> as the midpoint in <img src="https://render.githubusercontent.com/render/math?math=Z"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=z%27%20%5Cin%20Z%20s.t.%20Pr%28z%27%20%3C%3D%20Z%29%20%3D%200.5">) we could simply train <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> to change the value of <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20z%27%29"> a constant movement number <img src="https://render.githubusercontent.com/render/math?math=M"> greater for every training example <img src="https://render.githubusercontent.com/render/math?math=y%20%5Cin%20y_%7Btrain%7D"> that was greater than <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20z%27%29"> itself and the same constant number smaller for every training example that was smaller (remember the 2 pieces of fabric and the pins analogy). This would cause after enough iterations for the value of <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2Cz%27%29"> to settle in a position that divides in half all training examples when the total movement equals 0.

If instead of being <img src="https://render.githubusercontent.com/render/math?math=Z">'s midpoint <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29%20%5Cneq%200.5"> then the constant numbers of movement for greater and smaller samples have to be different.

Let's say that <img src="https://render.githubusercontent.com/render/math?math=a"> is the distance between <img src="https://render.githubusercontent.com/render/math?math=z%27"> and the smallest number in <img src="https://render.githubusercontent.com/render/math?math=Z"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D">, and <img src="https://render.githubusercontent.com/render/math?math=b"> the distance between <img src="https://render.githubusercontent.com/render/math?math=z%27"> and <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D">.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Barray%7D%7Bll%7D%0Aa%20%3D%20z%27%20-%20Z_%7Bmin%7D%5C%5C%0Ab%20%3D%20Z_%7Bmax%7D%20-%20z%27%0A%5Cend%7Barray%7D%0A">|



Since <img src="https://render.githubusercontent.com/render/math?math=a"> represents the amount of training examples we hope to find smaller than <img src="https://render.githubusercontent.com/render/math?math=z%27"> and <img src="https://render.githubusercontent.com/render/math?math=b"> the amount of training examples greater than <img src="https://render.githubusercontent.com/render/math?math=z%27"> we need 2 scalars <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> to satisfy the following equations:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20a%20%3D%20%5Cbeta%20b%20%5Ctag%7B8%7D%0A">|

These movement scalars will be the multipliers to be used with the constant movement <img src="https://render.githubusercontent.com/render/math?math=M"> on every observed point smaller and greater than <img src="https://render.githubusercontent.com/render/math?math=z%27"> respectively. This first equation assures that the total movement when <img src="https://render.githubusercontent.com/render/math?math=z%27"> is situated at <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D%20%2B%20a"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D%20-%20b"> will be 0.

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

Also as one of <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> or <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> tend to infinity the other one becomes a small number that is also impractical but for a different reason. The <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> at the edges are supposed to map to the edges of <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> and any quantity of movement into the opposite direction will result in <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmin%7D"> or <img src="https://render.githubusercontent.com/render/math?math=Z_%7Bmax%7D"> mapping to a greater or smaller point in <img src="https://render.githubusercontent.com/render/math?math=Y_x"> respectively. For this reason the <img src="https://render.githubusercontent.com/render/math?math=%5Calpha"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> for the <img src="https://render.githubusercontent.com/render/math?math=z_%7Bsamples%7D"> at the edges (i.e. <img src="https://render.githubusercontent.com/render/math?math=z%5B0%5D"> and <img src="https://render.githubusercontent.com/render/math?math=z%5BS%5D">) will be assigned a value of 0 for the one that pushes inward and a predefined hyperparameter <img src="https://render.githubusercontent.com/render/math?math=C%20%5Cin%20%5B0%2C%201%5D"> that can be adjusted to the model.

## Training the model

In order to train the neural network the <img src="https://render.githubusercontent.com/render/math?math=z-samples">, the set size <img src="https://render.githubusercontent.com/render/math?math=S"> is chosen depending on the desired accuracy and compute available. Having decided that, <img src="https://render.githubusercontent.com/render/math?math=Z"> must be defined. That is, the number of dimensions and its range must be chosen. Given <img src="https://render.githubusercontent.com/render/math?math=Z"> and the training level we can create the <img src="https://render.githubusercontent.com/render/math?math=z-samples"> set.

For example if <img src="https://render.githubusercontent.com/render/math?math=Z"> is 1-dimensional with range defined as <img src="https://render.githubusercontent.com/render/math?math=%5B10.0%2C%2020.0%5D"> and <img src="https://render.githubusercontent.com/render/math?math=S%20%3D%209">, we have that the *z-sample* set is <img src="https://render.githubusercontent.com/render/math?math=%5C%7Bz_%7B0%7D%20%2810.0%29%2C%20z_%7B1%7D%20%2811.25%29%2C%20z_%7B2%7D%20%2812.5%29%2C%20z_%7B3%7D%20%2813.75%29%2C%20z_%7B4%7D%20%2815.0%29%2C%20z_%7B5%7D%20%2816.25%29%2C%20z_%7B6%7D%20%2817.5%29%2C%20z_%7B7%7D%20%2818.75%29%2C%20z_%7B8%7D%20%2820.0%29%5C%7D">.

First we select a batch of data from the training data with size <img src="https://render.githubusercontent.com/render/math?math=n">, for every data point in the batch, we evaluate the current model on every *z-sample*. This gives us the prediction matrix:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Af_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7BS%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7BS%7D%29%5C%5C%0A...%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bn%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bn%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7Bn%7D%2C%20z_%7BS%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

For every data point <img src="https://render.githubusercontent.com/render/math?math=%28x_%7Bi%7D%2C%20y_%7Bi%7D%29"> in the batch, we take the output value <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> and compare it with every value of its corresponding row in the prediction matrix (i.e. <img src="https://render.githubusercontent.com/render/math?math=%5Bf_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7B1%7D%29%2C%20...%2C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7B8%7D%29%5D">). After determining if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is greater or smaller than each predicted value, we produce 2 values for every element in the matrix:

### Movement scalar
The scalar will be <img src="https://render.githubusercontent.com/render/math?math=%5Calpha_%7Bz-sample%7D"> if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is smaller than the prediction and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta_%7Bz-sample%7D"> if <img src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D"> is greater.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=s_%7Bi%2C%20z-sample%7D%20%3D%20%5Cleft%5C%7B%0A%5Cbegin%7Barray%7D%7Bll%7D%0A%5Calpha_%7Bz-sample%7D%20%26%20y_%7Bi%7D%20%3C%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%5C%5C%0A%5Cbeta_%7Bz-sample%7D%20%26%20y_%7Bi%7D%20%3E%20f_%7B%5Ctheta%7D%28x_%7Bi%7D%2C%20z_%7Bz-sample%7D%29%5C%5C%0A%5Cend%7Barray%7D%0A%5Cright.%0A">|

### Target value
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

The Mean Squared Error (MSE) loss function works to train the model using backpropagation and target values, but testing the model requires a different approach. Since both <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20Z%29"> and <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> are random variables, measuring the differences between their samplings is pointless. Because of this, the success of the model will be measured in 2 ways:

### Earth Movers Distance (EMD)

> In statistics, the **earth mover's distance** (**EMD**) is a measure of the distance between two probability distributions over a region *D*. In mathematics, this is known as the Wasserstein metric. Informally, if the distributions are interpreted as two different ways of piling up a certain amount of dirt over the region *D*, the EMD is the minimum cost of turning one pile into the other; where the cost is assumed to be amount of dirt moved times the distance by which it is moved. [wikipedia.org][EMD]

Using [EMD][EMD] we can obtain an indicator of how similar <img src="https://render.githubusercontent.com/render/math?math=Y_%7Bx%7D"> and <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%28x%2C%20Z%29"> are. It can be calculated by comparing every <img src="https://render.githubusercontent.com/render/math?math=x%2C%20y"> data point in the test data and prediction data sets and finding way to transform one into the other that requires the smallest total movement. What the EMD number tells us is the average amount of distance to transform every point in the predictions data set to the test data set.

On the example below you can see that the mean EMD is ~3.9 on a data set with a thickness of roughly 100. Because of the random nature of the data sets the EMD cannot be used as a literal error indicator, but it can be used as a progress indicator, that is to tell if the model improves with training.

|      |
| :-: |
|   <img src="images\fig5.png" alt="fig5" style="zoom:50%;" />   |
| **Fig 5** EMD testing. |

### Testing the training goals
###### Training goal 1

Ideally to test [Goal 1](#Goal-1) (i.e. <img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cwedge%20%5Cforall%20z%27%20%5Cin%20z_%7Bsamples%7D%3A%20arg%20%5C%20min_%7B%5Ctheta%7D%20%7CE_%7B%5Ctheta%7D%28z%27%29%7C">) we would evaluate <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> for a given <img src="https://render.githubusercontent.com/render/math?math=x"> and on every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> and then compare it to an arbitrary number of test data points having the same <img src="https://render.githubusercontent.com/render/math?math=x">. We would then proceed to count for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> the number of test data points smaller than it. With a vector of *smaller than counts* (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%27%29%29">) we could proceed to compare it with the canonical counts for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%27%29">) and measure the error. However in real life data sets this is not possible. Real life data sets will not likely have an arbitrary number of data points having the same <img src="https://render.githubusercontent.com/render/math?math=x"> (they will unlikely even have 2 data points with the same <img src="https://render.githubusercontent.com/render/math?math=x">) which means that we need to use a vicinity in <img src="https://render.githubusercontent.com/render/math?math=x"> (values <img src="https://render.githubusercontent.com/render/math?math=X"> that are close to an <img src="https://render.githubusercontent.com/render/math?math=x">) to test the goal.

We start by creating an ordering (an array of indices) <img src="https://render.githubusercontent.com/render/math?math=O%20%3D%20%5C%7Bo_%7B0%7D%2C%20o_%7B1%7D%2C%20...%2C%20o_%7Bm%7D%5C%7D"> that sorts all the elements in <img src="https://render.githubusercontent.com/render/math?math=X_%7Btest%7D"> (the <img src="https://render.githubusercontent.com/render/math?math=x"> inputs in the test data set). Then we select a substring of the array <img src="https://render.githubusercontent.com/render/math?math=O%27%20%3D%20%5C%7Bo_%7Bi%7D%2C%20o_%7Bi%2B1%7D%2C%20...%2C%20o_%7Bj%7D%5C%7D">.

Now we can evaluate <img src="https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D"> for every <img src="https://render.githubusercontent.com/render/math?math=x_%7Bo%27%7D%20%5Cmid%20o%27%20%5Cin%20O%27"> on every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> which gives us the matrix:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Af_%7B%5Ctheta%7D%28x_%7Bo_%7Bi%7D%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7Bi%7D%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7Bi%7D%7D%2C%20z_%7BS%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bo_%7Bi%2B1%7D%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7Bi%2B1%7D%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7Bi%2B1%7D%7D%2C%20z_%7BS%7D%29%5C%5C%0A...%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bo_%7Bj%7D%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7Bj%7D%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bo_%7Bj%7D%7D%2C%20z_%7BS%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

We then proceed to compare each row with the outputs <img src="https://render.githubusercontent.com/render/math?math=y_%7Bo%27%7D%20%5Cmid%20o%27%20%5Cin%20O%27">

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Ay_%7Bo_%7B0%7D%7D%5C%5C%0Ay_%7Bo_%7B1%7D%7D%5C%5C%0A...%5C%5C%0Ay_%7Bo_%7BV%7D%7D%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|

and create *smaller than counts* (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Y_%7Bx%7D%20%3C%3D%20f_%7B%5Ctheta%7D%28x%2C%20z%29%29">) which we can then compare with the canonical counts for every <img src="https://render.githubusercontent.com/render/math?math=z-sample"> (i.e. <img src="https://render.githubusercontent.com/render/math?math=P%28Z%20%3C%3D%20z%29">) to measure the error in the selected substring.

We will create a number of such substrings and call each error the local vicinity error located at the central element of the substring.

On the example below you can see that the goal 1 mean error is ~1.6%, this can be used as an error indicator for the model.

|      |
| :-: |
|   <img src="images\fig6.png" alt="fig6" style="zoom:50%;" />   |
| **Fig 6** Training goal 1 testing. |



###### Training goal 2

In order to test [Goal 2](#Goal-2) <img src="https://render.githubusercontent.com/render/math?math=%5Cforall%20x%20%5Cin%20X%20%5Cwedge%20%5Cforall%20z_%7B0%7D%2C%20z_%7B1%7D%20%5Cin%20Z%20%5C%20s.t.%20%5C%20z_%7B0%7D%20%3C%20z_%7B1%7D%3A%20f_%7B%5Ctheta%7D%28x%2C%20z_%7B0%7D%29%20%3C%20f_%7B%5Ctheta%7D%28x%2C%20z_%7B1%7D%29"> we select some random points in <img src="https://render.githubusercontent.com/render/math?math=X"> and a set of random points in <img src="https://render.githubusercontent.com/render/math?math=Z">, we run them in our model and get result matrix:

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0Af_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7B0%7D%2C%20z_%7Bn%7D%29%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7B1%7D%2C%20z_%7Bn%7D%29%5C%5C%0A...%5C%5C%0Af_%7B%5Ctheta%7D%28x_%7Bm%7D%2C%20z_%7B0%7D%29%2C%20f_%7B%5Ctheta%7D%28x_%7Bm%7D%2C%20z_%7B1%7D%29%2C%20...%20%2C%20f_%7B%5Ctheta%7D%28x_%7Bm%7D%2C%20z_%7Bn%7D%29%5C%5C%0A%5Cend%7Bbmatrix%7D%0A">|


From here it is trivial to check that each row is monotonically increasing. To increase the quality of the check we can increment the sizes of the test point set in <img src="https://render.githubusercontent.com/render/math?math=X"> and the test point set in <img src="https://render.githubusercontent.com/render/math?math=Z">.

## Back to delays

Now we can go back to the departure delays to arrival delays dataset, below you can see the MLE approach (**Fig 7a**) and the approach introduced in this article (**Fig7b**) side by side. As we saw before the MLE approach fails to capture the small imperfections obtaining a goal 1 error of 2.48% while the generic approach does a much better job with a 0.018% goal 1 error.


|      |      |
| ---- | ---- |
|   <img src="images\delay_prob_plots_res.gif" alt="fig7a" style="zoom:90%;" />   |   <img src="images\delay_gen_plots_res.gif" alt="fig7b" style="zoom:90%;" />   |
|   <img src="images\fig_delay_prob_tensorboard.png" alt="fig7a" style="zoom:30%;" />   |   <img src="images\fig_delay_gen_tensorboard.png" alt="fig7b" style="zoom:30%;" />   |
| **Fig 7a** Probabilistic model MLE approach. | **Fig 7b** Generic approach. |

## Experiments

The following are various experiments done on different datasets.

### <img src="https://render.githubusercontent.com/render/math?math=%5Clarge%20x%5E2"> plus gaussian noise

Let's start with a simple example. The function <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> with added gaussian noise. On the left panel you can see the training evolving over the course of 180 epochs. On the top left corner of this panel you can see the goal 1 error localized over <img src="https://render.githubusercontent.com/render/math?math=X">, at the end of the training you can see that the highest local error is around 2% and the global error is around 0.5%. On the top right corner of the same panel you can see the local Earth Mover's Distance (EMD). On the bottom left corner you can see a plot of the original test dataset (in blue) and the <img src="https://render.githubusercontent.com/render/math?math=z-lines"> (in orange), you can see how they progressively conform to the test data. On the bottom right you can see a plot of the original test dataset (in blue) and random predictions (with <img src="https://render.githubusercontent.com/render/math?math=z%20%5Csim%20Z">), you can see as the predicted results progressively represent the test data.

On the right panel, you can see a plot of the global goal 1 error (above) and global EMD values (below) as they change over the course of the training.

|      |      |
| ---- | ---- |
|   <img src="images\x2_normal_plots_res.gif" alt="fig8" style="zoom:80%;" />   |   <img src="images\fig_x2norm_tensorboard.png" alt="fig7" style="zoom:50%;" />   |
| **Fig 8** Training model to match <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> plus gaussian. |  |

### <img src="https://render.githubusercontent.com/render/math?math=%5Clarge%20a%20x%5E3%20%2B%20bx%5E2%20%2B%20cx%20%2B%20d"> plus truncated gaussian noise

This one is a bit more complicated. An order 3 polynomial with added truncated gaussian noise (that is a normal distribution clipped at specific points).

|      |      |
| ---- | ---- |
|   <img src="images\x3x2_trunc_plots_res.gif" alt="fig9" style="zoom:80%;" />   |   <img src="images\fig_x3x2trunc_tensorboard.png" alt="fig9" style="zoom:50%;" />   |
| **Fig 9** Training model to match <img src="https://render.githubusercontent.com/render/math?math=x%5E3%20%2B%20bx%5E2%20%2B%20cx%20%2B%20d"> plus truncated gaussian. |  |

### Double <img src="https://render.githubusercontent.com/render/math?math=%5Clarge%20sin%28x%29"> plus gaussian noise multiplied by sin(x)

This one is quite more interesting. 2 mirroring <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29"> functions with gaussian noise scaled by <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29"> itself.

| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=f%20%3D%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%0Asin%28x%29%20%2B%20%5Cmathcal%7BN%7D%20%2A%20sin%28x%29%20%26%20U%280%2C1%29%20%3C%3D%200.5%5C%5C%0A-sin%28x%29%20%2B%20%5Cmathcal%7BN%7D%20%2A%20sin%28x%29%20%26%20U%280%2C1%29%20%3E%200.5%5C%5C%0A%5Cend%7Barray%7D%5Cright.%0A">|



Notice how the model succeeds to represent the areas in the middle with lower densities.


|      |      |
| ---- | ---- |
|   <img src="images\sin_sin_plots_res.gif" alt="fig10" style="zoom:80%;" />   |   <img src="images\fig_sinsin_tensorboard.png" alt="fig10" style="zoom:50%;" />   |
| **Fig 10** Training model to match double <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29"> with added gaussian times <img src="https://render.githubusercontent.com/render/math?math=sin%28x%29">. |  |

### Branching function plus gaussian noise

This one experiments with branching paths. It starts with simple gaussian noise around <img src="https://render.githubusercontent.com/render/math?math=0">, then starts splitting it with equal probabilities over the course of various segments.


| |
|:-:|
|<img src="https://render.githubusercontent.com/render/math?math=f%20%3D%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%0A0.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B-4%2C%20-2%5D%5C%5C%0A1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B-2%2C%200%5D%20%26%5Cand%20%26U%280%2C1%29%20%3C%3D%200.5%5C%5C%0A-1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B-2%2C%200%5D%20%26%5Cand%20%260.5%20%3C%20U%280%2C1%29%5C%5C%0A0.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B0%2C%202%5D%20%26%5Cand%20%260.5%20%3C%20U%280%2C1%29%5C%5C%0A2.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B0%2C%202%5D%20%26%5Cand%20%260.25%20%3C%20U%280%2C1%29%20%3C%3D%200.5%5C%5C%0A-2.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B0%2C%202%5D%20%26%5Cand%20%26U%280%2C1%29%20%3C%3D%200.25%5C%5C%0A1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%26U%280%2C1%29%20%3C%3D%200.375%5C%5C%0A-1.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%260.375%20%3C%20U%280%2C1%29%20%3C%3D%200.75%5C%5C%0A3.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%260.75%20%3C%20U%280%2C1%29%20%3C%3D%200.875%5C%5C%0A-3.0%20%2B%20%5Cmathcal%7BN%7D%20%26%20%5Cforall%20x%20%5Cin%20%5B2%2C%204%5D%20%26%5Cand%20%260.875%20%3C%20U%280%2C1%29%5C%5C%0A%5Cend%7Barray%7D%5Cright.%0A">|


Despite the distribution not being continuous, the model does a reasonably good job of approximating it.

|      |      |
| ---- | ---- |
|   <img src="images\branch_norm_plots_res.gif" alt="fig11" style="zoom:80%;" />   |   <img src="images\fig_branchnorm_tensorboard.png" alt="fig11" style="zoom:50%;" />   |
| **Fig 11** Training model to match branching function plus gaussian. |  |

### <img src="https://render.githubusercontent.com/render/math?math=%5Clarge%20x_%7B0%7D%5E2%20%2B%20x_%7B1%7D%5E3"> plus absolute gaussian noise

The next example has 2 dimensions of input. <img src="https://render.githubusercontent.com/render/math?math=X_%7B0%7D"> (the first dimension) is <img src="https://render.githubusercontent.com/render/math?math=x%5E2"> and <img src="https://render.githubusercontent.com/render/math?math=X_%7B1%7D"> (the second dimension) is <img src="https://render.githubusercontent.com/render/math?math=x%5E3"> with added absolute gaussian noise. The display is slightly different, for the sake of space the <img src="https://render.githubusercontent.com/render/math?math=z-lines"> plot is omitted. As you can see there is a panel per dimension and as always an additional panel for goal 1 error and EMD error histories.

|      |
| ---- |
|<img src="images\x3_x2_absnormal_plots_res_0.gif" alt="fig12_0" style="zoom:80%;" />|
|<img src="images\x3_x2_absnormal_plots_res_1.gif" alt="fig12_1" style="zoom:80%;" />|
|<img src="images\fig_x3x2abs_tensorboard.png" alt="fig12" style="zoom:50%;" />|
| **Fig 12** Training model to match <img src="https://render.githubusercontent.com/render/math?math=x_%7B0%7D%5E2%20%2B%20x_%7B1%7D%5E3"> plus absolute gaussian. |

### California housing dataset

This experiment uses real data instead of generated one which proves the model's effectivity on real data. It is the classic [California housing dataset][CAL]. It has information from the 1990 California census with 8 input dimensions (Median Income, House Age, etc ...). Below you can see the plots of each dimension.

|      |
| ---- |
|<img src="images\california_housing_plots_res_0.gif" alt="fig13_0" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_1.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_2.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_3.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_4.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_5.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_6.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\california_housing_plots_res_7.gif" alt="fig13_1" style="zoom:66%;" />|
|<img src="images\fig_cal_tensorboard.png" alt="fig13" style="zoom:50%;" />|
| **Fig 13** Training model to match the California housing dataset. |



## Conclusion

The method presented allows to approximate the distributions of stochastic data sets to an arbitrary precision. The model is simple, fast to train and can be implemented with a vanilla feedforward neural network. Its ability to approximate any distribution across an input space makes it a valuable tool for any task that requires prediction.

## References

[UAT]: https://en.wikipedia.org/wiki/Universal_approximation_theorem
[EMD]: https://en.wikipedia.org/wiki/Earth_mover%27s_distance "Earth Mover's Distance"
[DEL]: https://www.kaggle.com/usdot/flight-delays	"2015 Flight Delays and Cancellations Dataset"
[MLE]: https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation/	"Logistic regression with maximum likelihood estimation"
[CAL]: http://lib.stat.cmu.edu/datasets/houses.zip "California Housing Dataset"