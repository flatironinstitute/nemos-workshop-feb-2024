r"""

# Generalized Linear Models (GLM): an Introduction

This tutorial aim to familiarize the neuroscience audience with the concept of GLM.
Readers will learn about the basics of GLM theory, and explore the concepts with simple synthetic data.

## What is a Generalized Linear Model (GLM)?

A Generalized Linear Model (GLM) is as a statistical model that maps inputs
$\bm{X}$ (e.g., auditory or visual stimuli, injected current, activity of neighboring neurons, etc.)
to some output $\bm{y}$ (e.g. neural activity).

*[Image: Encoding Model schematic]*

It specifies an encoding model $\mathbb{P}(\bm{y} | \bm{X})$ that quantifies the likelihood of observing the neural
activity $\bm{y}$ for a given input configuration $\bm{X}$. In particular, the $t$-th input sample
$\bm{x}_t = [x\_{t1}, ..., x\_{tk}]$ ($k$ is the number of features) is mapped into a firing rate
$\lambda_t$ by a linear-non-linear transformation,

$$
\lambda_t = f(\bm{x}_t \cdot \bm{w}) = f(x\_{t1} w\_1 + \cdots + x\_{tk} w\_k), \tag{1}
$$

where $\bm{w} = \begin{bmatrix} w_1 \\\ \vdots \\\ w_k \end{bmatrix}$
is a vector of weights that quantifies how much each input feature contributes to the neuron rate, and $f$ is a
non-linearity that converts the weighted sum of the input into rates.

!!! definition
    If the function $f$ is invertible, its inverse, denoted by $f^{-1}$, is commonly referred to as
    the **link function** in statistical literature.

Below an implementation of eqn (1),
"""

import jax.numpy as jnp

from numpy.typing import NDArray
from typing import Callable


# define a function for computing the rate
def compute_rate(weights: NDArray, x: NDArray, non_linearity: Callable=jnp.exp):
    """
    Compute the neuronal rate.

    Parameters
    ----------
    weights:
        Array of weights, shape (n_features, n_neurons)
    x:
        Array with the sampled input features, shape (n_samples, n_features)
    non_linearity:
        Function that transform the weighted inputs into rate.

    Returns
    -------
    :
        The predicted rate.

    """
    return non_linearity(x @ weights)

# %%
# The neural activity is described as a sample drawn from a probability distribution
# $\mathbb{P}(y | \bm{x}_t)$ with mean equal to the rate $\lambda_t$,
#
# $$
# \begin{cases}
# y \sim \mathbb{P}(y | \bm{x}_t) \\\
# \lambda_t = \mathbb{E}\_{\mathbb{P}}[y|\bm{x}_t].
# \end{cases}
# $$
#
# If we assume that the activity at different sample points is independent once we know the input, we obtain encoding
# model,
#
# $$
# \mathbb{P}(\bm{y} | \bm{X}) = \prod_{t=1}^{T}  \mathbb{P}(y_t | \bm{x}_t). \tag{2}
# $$
#
# Equation (2) implicitly depends on the model weights through the rate $\lambda_t$. When considered as a function of the
# weights, it is referred to as the model **likelihood**.
#
# !!! note
#     The mean alone is often insufficient to fully characterize a probability distribution.
#     For instance, a univariate Gaussian distribution is defined by two parameters: the mean and the variance.
#     Similarly, for most distributions in the exponential family utilized in Generalized Linear Models (GLMs),
#     additional parameters are needed. In practical applications involving common distributions such as
#     Gaussian, Inverse Gaussian, Bernoulli, and Poisson, the parameters defining the rate can be independently learned
#     from other statistics. This is not true in general; one may need to learn jointly the model parameters
#     for complex distributions.
#
# !!! warning
#     Some literature uses GLM referring to the *General Linear Models*.
#     This model class is different from the *Generaliz***ed** *linear models* defined here; The former are linear Gaussian
#     models, the latter, we will see, are non-linear models as defined in [REF Nelder and Wedderburn (1972)].
#
# ## The Poisson GLM
#
# The most commonly used GLM in neuroscience is the **Poisson GLM**.
# It is characterized by Poisson likelihood and a positive non-linearity $f$ (common choices are the
# exponential or soft-plus), that transform linear predictors into non-negative firing rates.
#
# $$
# \begin{aligned}
# \mathbb{P}(\bm{y} | \bm{X}) &= \prod_{t=1}^{T} \frac{\lambda_t^{y_t} \exp(-\lambda_t)}{y_t!} \\\
# &= \prod_{t=1}^{T} \frac{f(\bm{x}_t \cdot \bm{w})^{y_t} \exp(-f(\bm{x}_t \cdot \bm{w}))}{y_t!} \tag{3}
# \end{aligned}
# $$
#
# This is a convenient choice for modeling spike counts ($y_t \in \mathbb{N}_0$, i.e. is a positive
# integer) since,
#
# 1. The Poisson distribution models counting data.
# 2. It is fully specified by the rate.
# 3. It assumes that the variance of the counts is equal to the mean; this is a reasonable assumption, although
# not strictly true in real data [REF]. See [1] for how to achieve supra- and sub-Poisson variability.
#
# In the neuroscience literature, this model class is sometimes referred to as the Linear-Non-Linear Poisson model
# (LNP-model).
#
# For reason that will become clear later, see
# [here](#learning-the-glm-weights-by-maximum-likelihood-ml), we will implement a function for computing the logarithm
# of the likelihood, i.e. the log-likelihood.

from jax.scipy.special import gammaln

def poisson_log_likelihood(weights: NDArray, x: NDArray, y: NDArray, rate_func: Callable):
    """

    Parameters
    ----------
    weights:
        Array of weights, shape (n_features, n_neurons)
    x:
        Array with the sampled input features, shape (n_samples, n_features)
    y:
        Array of counts, shape (n_samples, n_neurons)
    rate_func

    Returns
    -------

    """
    rate = rate_func(weights, x)
    # compute the normalization constant log(y!)
    norm_const = gammaln(y + 1).sum()
    return jnp.sum(y * jnp.log(rate) - rate) - norm_const

# %%
# ### A Simple Example
# To make things more concrete, let's immagine to model the rate of a neuron in response to the visual
# contrast of an image. Assume that input contrast is a one-dimensional array that assumes values between 0 and 1.
# Plot the rate as a function of the weights for a range of values.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
# generate 100 sample of random contrasts
n_samples = 100
n_features = 1

# generate equi-spaced contrast and weights and add the feature dimension
contrast = np.linspace(0, 1, n_samples)[:, None]
weights = np.linspace(0, 1, 4)[:, None]

plt.figure()
plt.title("Tuning Curve")
for w in weights:
    p, = plt.plot(contrast, compute_rate(w, contrast), label=f"weight: {w[0]:.2f}")
plt.ylabel("rate [spike/bin]")
plt.xlabel("contrast")
plt.legend()

# %%
# Let's see how to add a predefined offset.

# set offset
offset_weight = np.log(0.5)

# add an the offset feature (a constant)
features = np.hstack((np.ones((n_samples, 1)), contrast))
# add the weight specifying the offset
weights_with_offset = np.hstack((offset_weight * np.ones((4, 1)), weights))

# plot the new tuning function
plt.figure()
plt.title("Tuning Curve - Offset")
for w in weights_with_offset:
    plt.plot(contrast, compute_rate(w, features), label=f"weight: {w[1]:.2f}")
plt.ylabel("rate [spike/bin]")
plt.xlabel("contrast")
plt.legend()

# %%
# Let's now generate some synthetic data and plot the log-likelihood as a function of the weights.
n_samples = 100

# define some ground truth weights
weights_true = np.array([offset_weight, 0.2])

# generate some random constrast
contrast = np.random.uniform(size=(n_samples, 1))

# add the offset to the predictors
features = np.hstack((np.ones((n_samples, 1)), contrast))
counts = np.random.poisson(compute_rate(weights_true, features))

# define a range of weights
weights = np.linspace(-0.6, 1, 100)
# add true offset for simplicity
weights = np.hstack((offset_weight * np.ones((weights.shape[0], 1)), weights[:,None]))

# compute the likelihood of each weights
log_like = [poisson_log_likelihood(w, features, counts, rate_func=compute_rate) for w in weights]

plt.figure()
plt.plot(weights[:, 1], log_like)
ymin, ymax = plt.ylim()
plt.vlines(weights_true[1], ymin, ymax, "r", label="true weights")
plt.vlines(weights[np.argmax(log_like), 1], ymin, ymax, "k", label="maximum-likelihood")
plt.ylim(ymin, ymax)
plt.legend()
plt.ylabel("log-likelihood")
plt.xlabel("weight")

# %%
# ## Learning The GLM Weights by Maximum Likelihood (ML)
#
# In general, we are provided with paired observations $(y_t, \bm{x}_t)$, for samples $t=1,...,T$,
# where $y_t$ are the spike counts and $\bm{x}_t$ are our input features.
# Evaluating the encoding map (1) at the observations gives us a function of the weights, referred
# to as the model likelihood.  The ML estimator for the model parameters is the set of weights that
# maximizes the (log-)likelihood,
#
# $$
# \begin{aligned}
# \hat{\bm{w}} &= \argmax_{\bm{w}} \left(\log{ \mathbb{P}(\bm{y} | \bm{X}; \bm{w}) }\right) \\\
# &= \argmax_{\bm{w}} \sum_{t=1}^{T} {y_t} \log f(\bm{x}_t \cdot \bm{w}) -f(\bm{x}_t \cdot \bm{w}).
# \end{aligned}
# $$
#
# !!! note
#     - The logarithm is a monotonic function, therefore the log-likelihood and the likelihood are maximized by
#     the same weights $\hat{\bm{w}}$.
#     - The log-likelihood is numerically better behaved than the likelihood; A large product of small terms
#     (as in the likelihood) is likely to incur in numerical precision issues. This is why the log-likelihood is
#     maximized instead.
#     - In the second equation we omitted the $-\log(y_t!)$ term since it is constant in $\bm{w}$.
#
# The maximum likelihood estimator benefit has some nice properties:
#
#   - **Unbiasedness**: The expected value of the ML estimator are the true weights
#   - **Consistency**: Converges to the true weights in the limit of infinite adata
#   - **Efficiency**: Achieves minimum possible asymptotic error.
#
#  On top of that, for convex and monotonic non-linearities $f$ the log-likelihood of the GLM is convex (
#  see the figure above for example); this guarantees an extra desirable property:
#
#  - **Uniqueness**: the ML weights exists and are unique.
#
# In other words, learning the ML weights is a convex, differentiable optimization problem that can be solved
# by standard methods (gradient descent, Newton and quasi-Newton methods).
# Here we will show how to solve for the ML parameters using GradientDescent `jaxopt`, which computes the
# gradient of the likelihood for us through auto-differentiation.
import jax
import jaxopt

# enable float64
jax.config.update("jax_enable_x64", True)

jax_compute_rate = lambda w, x: compute_rate(w,x,jax.numpy.exp)
func = lambda w: -poisson_log_likelihood(w, features, counts, jax_compute_rate)
initial_weights = np.array([np.log(counts.mean()), 0])
ml_weights, state = jaxopt.BFGS(func, jit=True).run(initial_weights)

print("ML weights:", ml_weights)

# We can assess numerically some of the properties listed above. For example, for the unbiasedness
# we can draw multiple sample pairs $(\bm{x}, \bm{y}), and compute the empirical mean of the ML estimator
# in each draw,

n_draw = 100
ml_weights_list = []
for i in range(n_draw):
    f = np.hstack((np.ones((n_samples, 1)), contrast))
    c = np.random.poisson(compute_rate(weights_true, features))
    func = lambda w: -poisson_log_likelihood(w, f, c, jax_compute_rate)
    ml_weights_list.append(jaxopt.LBFGS(func, jit=True).run(initial_weights)[0])

weights = np.vstack(ml_weights_list).T

plt.figure(figsize=(7, 3))
plt.suptitle("Unbiasedness")
plt.subplot(121)
plt.scatter(*weights, color="Gray", alpha=0.5, s=10)
plt.plot(*weights.mean(axis=1), "ok", label="average ML")
plt.plot(weights_true[0], weights_true[1],"or", label="true")
plt.xlabel("intercept")
plt.ylabel("weight")
plt.legend()

plt.subplot(122)
error = [np.linalg.norm(weights[:, :i].mean(axis=1) - weights_true) for i in range(1, n_draw+1)]
plt.plot(range(1, n_draw+1), error)
plt.xlabel("number of draws")
plt.ylabel("error")
plt.tight_layout()

# %%
# Similarly, we can check for the consistency

sample_sizes = 10**np.arange(2, 5)
error = []
for n_samples in sample_sizes:
    contrast = np.random.uniform(size=(n_samples, 1))
    # add the offset to the predictors
    features = np.hstack((np.ones((n_samples, 1)), contrast))
    counts = np.random.poisson(compute_rate(weights_true, features))
    func = lambda w: -poisson_log_likelihood(w, features, counts, jax_compute_rate)
    ml_weights = jaxopt.LBFGS(func, jit=True).run(initial_weights)[0]
    error.append(np.linalg.norm(weights_true - ml_weights))

fig, ax = plt.subplots(1,1)
plt.plot(sample_sizes, error, "-ok")
ax.set_xscale("log")
plt.title("Consistency")
plt.ylabel("error")
plt.xlabel("number of samples")

# %%
# ## Fit with NeMoS
# NeMoS implements of the Poisson-GLM abstract away all the details. The default non-linearity is exponential
# and one can fit the GLM for the section before with the following lines of code

import nemos as nmo

model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("BFGS", dict(tol=10**-12)))
# the input must be of shape (n_samples, n_neurons, n_features)
# the counts must be of shape (n_samples, n_neurons)
# the offset is fit automatically
contrast = features[:, None, 1:]
print(f"(n_samples, n_neurons, n_features) = {contrast.shape}")
model.fit(contrast, counts[:, None])


print("nemos weights:      ", model.coef_[0, 0])
print("custom ML estimate: ", ml_weights[1])

# %%
# ## References
#
# 1. Weber, A. I., & Pillow, J. W. (2017). [Capturing the dynamical repertoire of single neurons with generalized linear
#   models](https://pillowlab.princeton.edu/pubs/Weber17_IzhikevichGLM_NC.pdf). Neural computation, 29(12), 3260-3289.

