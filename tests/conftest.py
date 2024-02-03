import jax
import jax.numpy as jnp
import numpy as np
import pytest

from workshop_utils.model import GLM
import nemos as nmo


@pytest.fixture
def poissonGLM_model_instantiation():
    """Set up a Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the tests data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 5))
    b_true = np.array(0.)
    w_true = np.random.normal(size=(5,))
    observation_model = nmo.observation_models.PoissonObservations(jnp.exp)
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = GLM(observation_model, regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


@pytest.fixture
def poissonGLM_model_instantiation_pytree(poissonGLM_model_instantiation):
    """Set up a Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = poissonGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1]
    )
    model_tree = GLM(model.observation_model, model.regularizer)
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate
