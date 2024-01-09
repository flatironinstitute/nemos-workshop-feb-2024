r"""
# What is a Generalized Linear Model (GLM)?

!!! warning
    Some literature uses GLM to refer General Linear Models.
    This model class is different from the Generaliz**ed** linear models defined here; The former are linear Gaussian
    models, the latter are non-linear models as defined in [REF].

A Generalized Linear Model (GLM) is as a statistical model that maps some inputs
$\bm{X}$ (i.e. auditory or visual stimuli, injected current, the activity of neighboring neurons...)
to neural activity $\bm{y}$.
It specifies an encoding map $\mathbb{P}(\bm{y} | \bm{X})$ that describes the probability of
observing neural activity given an input. The $t$-th input sample $\bm{x}_t = [x\_{t1}, ..., x\_{tk}]$
($k$ is the number of input features) is mapped into a firing rate $\lambda_t$ by a linear-non-linear transformation,

$$
\lambda_t = f(\bm{x}_t \cdot \bm{w}) = f(x\_{t1} w\_1 + \cdots + x\_{tk} w\_k),
$$

where $\bm{w} = \begin{bmatrix} w_1 \\\ \vdots \\\ w_k \end{bmatrix}$
is a vector of weights that quantifies how much each input feature contributes to the neuron rate, and $f$ is
non-linearity that converts the weighted sum of the input into a rate.

*[Image: Encoding Map schematic]*

The neural activity is described as a sample drawn from a probability distribution
$\mathbb{P}(y | \bm{x}_t)$ with mean equal to the rate $\lambda_t$,

$$
\begin{cases}
y \sim \mathbb{P}(y | \bm{x}_t) \\\
\lambda_t = \mathbb{E}\_{\mathbb{P}}[y|\bm{x}_t].
\end{cases}
$$

If we assume that the neural activity at different sample points is independent for known inputs, we obtain encoding
map,

 $$
 \mathbb{P}(\bm{y} | \bm{X}) = \prod_{t=1}^{T}  \mathbb{P}(y_t | \bm{x}_t).
 $$


!!! note
    The mean alone is often insufficient to fully characterize a probability distribution.
    For instance, a univariate Gaussian distribution is defined by two parameters: the mean and the variance.
    Similarly, for most distributions in the exponential family utilized in Generalized Linear Models (GLMs),
    additional parameters are needed. In practical applications involving common distributions such as
    Gaussian, Inverse Gaussian, Bernoulli, and Poisson, the parameters defining the rate can be independently learned
    from other statistics. This is not true in general; one may need to learn jointly the model parameters
    for complex distributions.

Currently, **nemos** implements the **Poisson GLM**, where the encoding map takes the form,

$$
\begin{aligned}
\mathbb{P}(\bm{y} | \bm{X}) &= \prod_{t=1}^{T} \frac{\lambda_t^{y_t} \exp(-\lambda_t)}{y_t!} \\\
&= \prod_{t=1}^{T} \frac{f(\bm{x}_t \cdot \bm{w})^{y_t} \exp(-f(\bm{x}_t \cdot \bm{w}))}{y_t!} \tag{1}
\end{aligned}
$$

This is a natural and a convenient choice for modeling spike counts ($y_t \in \mathbb{N}_0$, i.e. is a positive
integer) since,

1. The Poisson distribution models counting data.
2. It is fully specified by the rate.
3. It assumes that the variance of the counts is equal to the mean; this is a reasonable assumption, although
not strictly true in real data [REF].

!!! note
    If the function $f$ is invertible, its inverse, denoted by $f^{-1}$, is commonly referred to as
    the **link function** in statistical literature.

## Setting up a Poisson GLM

*Work out minimal example of a poisson glm for predicting counts given an injected current.*

# Learning The GLM Weights by Maximum Likelihood (ML)

In general, we are provided with paired observations $(y_t, \bm{x}_t)$, for samples $t=1,...,T$,
where $y_t$ are the spike counts and $\bm{x}_t$ are some simultaneously recorded input features. If we evaluate
the encoding map (1) at the observations, we are left with a function of the weights, which is referred to as the
model likelihood. The ML estimator for the model weights is the set of weights that maximizes the (log-)likelihood,

$$
\begin{aligned}
\hat{\bm{w}} &= \argmax_{\bm{w}} \left(\log{ \mathbb{P}(\bm{y} | \bm{X}; \bm{w}) }\right) \\\
&= \argmax_{\bm{w}} \sum_{t=1}^{T} {y_t} \log f(\bm{x}_t \cdot \bm{w}) -f(\bm{x}_t \cdot \bm{w}).
\end{aligned}
$$

!!! note
    - The logarithm is a monotonic function, therefore the log-likelihood and the likelihood are maximized by
    the same weights $\hat{\bm{w}}$.
    - The log-likelihood is numerically better behaved than the likelihood; A large product of small terms
    (as in the likelihood) is likely to incur in numerical precision issues. This is why the log-likelihood is
    maximized instead.
    - In the second equation we omitted the $-\log(y_t!)$ term since it is constant in $\bm{w}$.
    - For convex, monotonic functions, the log-likelihood is convex, i.e. the optimal weights exists and are
    unique.

## Learning the ML weights for the injected current example
*Use the instantaneous injected current as a single scalar input; use nemos to learn the ML parameters (intercept and
weight), check the model test-set fit accuracy.*


## Adding Temporal Effects
*Use the past history of the injected current as a predictor. Show that this is equivalent to
convolve the weights with the current; the convolution can be represented as matrix vector product;
show that for fine time resolution and long history it over-fits (too many params); introduce the basis as a way to
reduce the parameter-space dimension and smoothen the history effect. show if the basis increase the
fit accuracy*

"""
