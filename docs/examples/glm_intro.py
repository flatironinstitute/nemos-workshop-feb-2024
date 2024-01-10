r"""

# Generalized Linear Models (GLM): an Introduction

This tutorial aim to familiarize the neurosceince audience with the concept of GLM. The user will learn about
the fundamentals of the GLM theory, and apply the concepts to the analysis of an intracellularly recorded neuron.


## What is a Generalized Linear Model (GLM)?

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


## Setting up the Poisson GLM

### Dataset
In this tutorial we will focus on a dataset from the Allen Cell Types data set.
Our dataset consists of a whole cell current clamp recordings of an individual neurons.
The neuron is stimulated with an injected current that follows a varieties of protocols: short and long square waves,
ramps, withe noise, impulses.

The variable we are going to be working with will be the following:
 - trial_interval_set: a dictionary containing the start and end time of each trial in second. Dictionary keys
 will refer to the stimulation protocol used for the trial

The start and end of each trial in seconds is stored in a pynapple IntervaSet object.
Metadata about each trial (including the stimulation protocol used) will be stored in the `sweep_metadata` dictionary.
Spike times will be stored as a pynapple Ts (time series) object.

"""
import numpy as np
import matplotlib.pyplot as plt
from utils.load_allen_utils import load_to_pynapple

# load the data
(trial_interval_set,
 injected_current,
 response,
 spike_times,
 sweep_metadata) = load_to_pynapple("/Users/ebalzani/Code/fit_from_allensdk/cell_types/specimen_478498617/ephys.nwb")

# metadata for a trial
stim_type = "Noise 2"
sweep_num = 49
print(f"Stimulation protocol: {stim_type}\nTrial: {sweep_num}")
for key, values in sweep_metadata[stim_type][sweep_num].items():
    print(f"\t - {key}: {values}")

# # plot the trial
trial_index = sweep_metadata[stim_type][sweep_num]["trial_index"]
trial_interval = trial_interval_set[stim_type][trial_index: trial_index+1]
binned_current = injected_current.bin_average(bin_size=0.001, ep=trial_interval) * 10**12
plt.figure()
plt.title(f"Sweep {sweep_num} - {stim_type}")
plt.plot(binned_current, label="injected current")
plt.vlines(spike_times[1].restrict(trial_interval).t, 0, 0.1 * np.max(binned_current), 'k', label="spikes")
plt.legend()
plt.xlabel('time [sec]', fontsize=10)
plt.ylabel('current [pA]', fontsize=10)
plt.tight_layout()

# %%
# We can plot the explore our dataset recording by plotting the time-course of the current injection
# overlapped to the evoked spikes.

# bin the injected current at 1ms resolution
# and convert from Ampere to pico-Ampere
binned_current = injected_current.bin_average(bin_size=0.001) * 10**12

fig, ax = plt.subplots(figsize=(10, 3.5))
plt.plot(binned_current, label="injected current")
plt.vlines(spike_times[1].t, 0, 0.1 * np.max(binned_current), 'k', label="spikes")
ylim = plt.ylim()
for start, end in spike_times.time_support.values:
    rect = plt.Rectangle(
        xy=(start, 0),
        width=end - start,
        height=ylim[1],
        alpha=0.2,
        facecolor="grey",
        label="trial" if start == 0 else None
    )
    ax.add_patch(rect)
plt.xlabel('time [sec]', fontsize=10)
plt.ylabel('current [pA]', fontsize=10)
plt.legend()
plt.tight_layout()

# %%
# *Work out minimal example of a poisson glm for predicting counts given an injected current.*
#
# # Learning The GLM Weights by Maximum Likelihood (ML)
#
# In general, we are provided with paired observations $(y_t, \bm{x}_t)$, for samples $t=1,...,T$,
# where $y_t$ are the spike counts and $\bm{x}_t$ are some simultaneously recorded input features. If we evaluate
# the encoding map (1) at the observations, we are left with a function of the weights, which is referred to as the
# model likelihood. The ML estimator for the model weights is the set of weights that maximizes the (log-)likelihood,
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
#     - For convex, monotonic functions, the log-likelihood is convex, i.e. the optimal weights exists and are
#     unique.
#
# ## Learning the ML weights for the injected current example
# *Use the instantaneous injected current as a single scalar input; use nemos to learn the ML parameters (intercept and
# weight), check the model test-set fit accuracy.*
#
#
# ## Adding Temporal Effects
# *Use the past history of the injected current as a predictor. Show that this is equivalent to
# convolve the weights with the current; the convolution can be represented as matrix vector product;
# show that for fine time resolution and long history it over-fits (too many params); introduce the basis as a way to
# reduce the parameter-space dimension and smoothen the history effect. show if the basis increase the
# fit accuracy*
#
# ## References
#
# 1. Weber, A. I., & Pillow, J. W. (2017). [Capturing the dynamical repertoire of single neurons with generalized linear
#   models](https://pillowlab.princeton.edu/pubs/Weber17_IzhikevichGLM_NC.pdf). Neural computation, 29(12), 3260-3289.

