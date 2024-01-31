from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import numpy as np
import pytest

from nemos.pytrees import FeaturePytree

import nemos as nmo
import jax


class TestGLM:
    """
    Unit tests for the PoissonGLM class.
    """
    #######################
    # Test model.fit
    #######################
    @pytest.mark.parametrize(
        "n_params, expectation",
        [
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (1, pytest.raises(ValueError, match="Params must have length two.")),
            (2, does_not_raise()),
            (3, pytest.raises(ValueError, match="Params must have length two.")),
        ],
    )
    def test_fit_param_length(
        self, n_params, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize(
        "add_entry, add_to, expectation",
        [
            (0, "X", does_not_raise()),
            (np.nan, "X", pytest.raises(ValueError, match="Input .+ contains")),
            (np.inf, "X", pytest.raises(ValueError, match="Input .+ contains")),
            (0, "y", does_not_raise()),
            (np.nan, "y", pytest.raises(ValueError, match="Input .+ contains")),
            (np.inf, "y", pytest.raises(ValueError, match="Input .+ contains")),
        ],
    )
    def test_fit_param_values(
        self, add_entry, add_to, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with altered X or y values. Ensure the method raises exceptions for NaN or Inf values.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if add_to == "X":
            # get an index to be edited
            idx = np.unravel_index(np.random.choice(X.size), X.shape)
            X[idx] = add_entry
        elif add_to == "y":
            idx = np.unravel_index(np.random.choice(y.size), y.shape)
            y = np.asarray(y, dtype=np.float32)
            y[idx] = add_entry
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "dim_weights, expectation",
        [
            (0, pytest.raises(ValueError, match=r"Inconsistent number of features")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"params\[0\] must be an array or .* of shape \(n_features, \)")),
            (3, pytest.raises(ValueError, match=r"params\[0\] must be an array or .* of shape \(n_features, \)")),
        ],
    )
    def test_fit_weights_dimensionality(
        self, dim_weights, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_features = X.shape
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros((1, n_features))
        else:
            init_w = jnp.zeros((1, n_features) + (1,) * (dim_weights - 2))
        with expectation:
            model.fit(X, y, init_params=(init_w, true_params[1]))

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match=r"params\[1\] must be a scalar")),
            (2, pytest.raises(ValueError, match=r"params\[1\] must be a scalar")),
            (3, pytest.raises(ValueError, match=r"params\[1\] must be a scalar")),
        ],
    )
    def test_fit_intercepts_dimensionality(
        self, dim_intercepts, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_features = X.shape
        init_b = jnp.zeros((1,) * dim_intercepts)
        init_w = jnp.zeros(n_features)
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize(
        "init_params, expectation",
        [
            ([jnp.zeros((5, )), jnp.array(0.)], does_not_raise()),
            (dict(p1=jnp.zeros((1, 5)), p2=jnp.zeros((1,))), pytest.raises(KeyError)),
            ((dict(p1=jnp.zeros((5, )), p2=jnp.zeros((1))), jnp.array(0.)), pytest.raises(TypeError, match=r"X and params\[0\] must be the same type")),
            ((FeaturePytree(p1=jnp.zeros((5, )), p2=jnp.zeros((5, ))), jnp.array(0.)), pytest.raises(TypeError, match=r"X and params\[0\] must be the same type")),
            ([jnp.zeros((5, )), ""], pytest.raises(ValueError, match="could not convert string to float")),
        ],
    )
    def test_fit_init_params_type(
        self, init_params, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match=r"params\[1\] must be a scalar")),
        ],
    )
    def test_fit_n_neuron_match_baseline_rate(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method ensuring The number of neurons in the baseline rate matches the expected number.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        init_b = jnp.squeeze(jnp.zeros((1 + delta_n_neuron,)))
        with expectation:
            model.fit(X, y, init_params=(true_params[0], init_b))

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_fit_x_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_dim == -1:
            X = np.zeros((X.shape[0], ))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="axis 1 is out of bounds for array of dimension 1")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be one-dimensional")),
        ],
    )
    def test_fit_y_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_dim == -1:
            y = np.zeros([])
        elif delta_dim == 1:
            y = np.zeros((X.shape[0], X.shape[1]))
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_fit_n_feature_consistency_weights(
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        init_w = jnp.zeros(X.shape[1] + delta_n_features)
        init_b = jnp.array(0.)
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_fit_n_feature_consistency_x(
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (-1, pytest.raises(ValueError, match="The number of time-points in X and y")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="The number of time-points in X and y")),
        ],
    )
    def test_fit_time_points_x(
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (-1, pytest.raises(ValueError, match="The number of time-points in X and y")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="The number of time-points in X and y")),
        ],
    )
    def test_fit_time_points_y(
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.fit(X, y, init_params=true_params)

    def test_fit_pytree_equivalence(self, poissonGLM_model_instantiation,
                                    poissonGLM_model_instantiation_pytree):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        X_tree, _, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        # fit both models
        model.fit(X, y, init_params=true_params)
        model_tree.fit(X_tree, y, init_params=true_params_tree)

        # get the flat parameters
        flat_coef = np.concatenate(jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=-1)

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)

    def test_score_array(self, poissonGLM_model_instantiation):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.fit(X, y, init_params=true_params)
        model.score(X, y)

    def test_score_tree(self, poissonGLM_model_instantiation_pytree):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        model_tree.fit(X_tree, y, init_params=true_params_tree)
        model_tree.score(X_tree, y)

    def test_predict_array(self, poissonGLM_model_instantiation):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.fit(X, y, init_params=true_params)
        model.predict(X)

    def test_predict_tree(self, poissonGLM_model_instantiation_pytree):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        model_tree.fit(X_tree, y, init_params=true_params_tree)
        model_tree.predict(X_tree)

    def test_output_type_consistency_tree_fit(self, poissonGLM_model_instantiation_pytree):
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        model_tree.fit(X_tree, y, init_params=true_params_tree)
        assert(isinstance(X_tree, nmo.pytrees.FeaturePytree))

    def test_output_type_consistency_tree_score(self, poissonGLM_model_instantiation_pytree):
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        model_tree.fit(X_tree, y, init_params=true_params_tree)
        model_tree.score(X_tree, y)
        assert (isinstance(X_tree, nmo.pytrees.FeaturePytree))

    def test_output_type_consistency_tree_predict(self, poissonGLM_model_instantiation_pytree):
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        model_tree.fit(X_tree, y, init_params=true_params_tree)
        model_tree.predict(X_tree)
        assert (isinstance(X_tree, nmo.pytrees.FeaturePytree))

    def test_predict_reset_params(self, poissonGLM_model_instantiation_pytree):
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        X_tree["input_1"]
        try:
            model_tree.fit(X_tree, y, init_params=true_params_tree)
            X_tree["input_1"] = X_tree["input_1"][:,:1]
            model_tree.predict(X_tree)
            raise ValueError("No exception raised")
        except:
            # check that the coeffs have been squeezed back even if
            # predict did not go through
            assert nmo.utils.pytree_map_and_reduce(lambda x, y: jax.numpy.array(x.shape) == np.array(y.shape), any,
                                                   model_tree.coef_, true_params_tree[0])

    def test_score_reset_params(self, poissonGLM_model_instantiation_pytree):
        X_tree, y, model_tree, true_params_tree, _ = poissonGLM_model_instantiation_pytree
        X_tree["input_1"]
        try:
            model_tree.fit(X_tree, y, init_params=true_params_tree)
            X_tree["input_1"] = X_tree["input_1"][:,:1]
            model_tree.score(X_tree, y)
            raise ValueError("No exception raised")
        except:
            # check that the coeffs have been squeezed back even if
            # predict did not go through
            assert nmo.utils.pytree_map_and_reduce(lambda x, y: jax.numpy.array(x.shape) == np.array(y.shape), any,
                                                   model_tree.coef_, true_params_tree[0])