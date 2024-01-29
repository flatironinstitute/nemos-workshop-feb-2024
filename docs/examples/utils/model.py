import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import nemos as nmo
from nemos.base_class import DESIGN_INPUT_TYPE
from typing import Optional, Tuple, Union, Literal
import jax


class GLM(nmo.glm.GLM):

    def fit(
            self,
            X: DESIGN_INPUT_TYPE,
            y: ArrayLike,
            init_params: Optional[
            Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]
            ] = None
    ) -> nmo.glm.GLM:
        """Fit GLM to neural activity.

            Fit and store the model parameters as attributes
            ``coef_`` and ``coef_``.

            Parameters
            ----------
            X :
                Predictors, array of shape (n_time_bins, n_features) or pytree of same.
            y :
                Target neural activity arranged in a matrix, shape (n_time_bins, ).
            init_params :
                2-tuple of initial parameter values: (coefficients, intercepts). If
                None, we initialize coefficients with zeros, intercepts with the
                log of the mean neural activity. coefficients is an array of shape
                (n_features, ) or pytree of same, intercepts is an array
                of shape (1,)

            Raises
            ------
            ValueError
                - If `init_params` is not of length two.
                - If dimensionality of `init_params` are not correct.
                - If `X` is not two-dimensional.
                - If `y` is not one-dimensional.
                - If solver returns at least one NaN parameter, which means it found
                  an invalid solution. Try tuning optimization hyperparameters.
            TypeError
                - If `init_params` are not array-like
                - If `init_params[i]` cannot be converted to jnp.ndarray for all i

            """
        if init_params is not None:
            if len(init_params) != 2:
                raise ValueError("Params must have length two.")
            init_params = (jax.tree_map(lambda x: np.expand_dims(x, axis=0), init_params[0]), init_params[1])
        super().fit(jnp.expand_dims(X, axis=1), jnp.expand_dims(y, axis=1), init_params=init_params)
        self.coef_ = jax.tree_map(np.squeeze, self.coef_)
        return self

    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        self.coef_ = jax.tree_map(lambda x: np.expand_dims(x, axis=0), self.coef_)
        rate = super().predict(jnp.expand_dims(X, axis=1))
        self.coef_ = jax.tree_map(np.squeeze, self.coef_)
        return rate

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "pseudo-r2-McFadden",
    ) -> jnp.ndarray:
        self.coef_ = jax.tree_map(lambda x: np.expand_dims(x, axis=0), self.coef_)
        score = super().score(jnp.expand_dims(X, axis=1), jnp.expand_dims(y, axis=1), score_type=score_type)
        self.coef_ = jax.tree_map(np.squeeze, self.coef_)
        return score

    @staticmethod
    def _check_input_dimensionality(
            X: Optional[DESIGN_INPUT_TYPE] = None,
            y: Optional[jnp.ndarray] = None,
    ):
        if not (y is None):
            if y.ndim != 2:
                raise ValueError(
                    "y must be one-dimensional, with shape (n_timebins, )"
                )
        if not (X is None):
            if nmo.utils.pytree_map_and_reduce(lambda x: x.ndim != 3, any, X):
                raise ValueError(
                    "X must be two-dimensional, with shape (n_timebins, n_features) or pytree of the same"
                )

    @staticmethod
    def _check_input_and_params_consistency(
            params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
            X: Optional[DESIGN_INPUT_TYPE] = None,
            y: Optional[jnp.ndarray] = None,
    ):
        """Validate the number of neurons in model parameters and input arguments.

        Raises
        ------
        ValueError
            - if the number of neurons is inconsistent across the model
              parameters (`params`) and any additional inputs (`X` or `y` when
              provided).
            - if the number of features is inconsistent between params[0] and X
              (when provided).

        """
        if X is not None:
            X_structure = jax.tree_util.tree_structure(X)
            params_structure = jax.tree_util.tree_structure(params[0])
            if X_structure != params_structure:
                raise TypeError(
                    f"X and params[0] must be the same type, but X is {type(X)} and "
                    f"params[0] is {type(params[0])}"
                )
            if nmo.utils.pytree_map_and_reduce(
                    lambda p, x: p.shape[1] != x.shape[2], any, params[0], X
            ):
                raise ValueError(
                    "Inconsistent number of features. "
                    f"spike basis coefficients has {jax.tree_map(lambda p: p.shape[1], params[0])} features, "
                    f"X has {jax.tree_map(lambda x: x.shape[2], X)} features instead!"
                )

    @staticmethod
    def _check_and_convert_params(
            params: Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike],
            data_type: Optional[jnp.dtype] = None,
    ) -> Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters and data.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        if not hasattr(params, "__len__") or len(params) != 2:
            raise ValueError("Params must have length two.")

        try:
            params = jax.tree_map(lambda x: jnp.asarray(x, dtype=data_type), params)
        except (ValueError, TypeError):
            raise TypeError(
                "Initial parameters must be array-like objects (or pytrees of array-like objects) "
                "with numeric data-type!"
            )

        if nmo.utils.pytree_map_and_reduce(lambda x: x.ndim != 2, any, params[0]):
            raise ValueError(
                "params[0] must be an array or nemos.pytree.FeaturePytree with array leaves "
                "of shape (n_features, )."
            )

        if params[1].ndim != 1:
            raise ValueError(
                "params[1] must be one-dimensional but "
                f"params[1] has {params[1].ndim} dimensions!"
            )

        if params[1].shape[0] != 1:
            raise ValueError(f"Exactly one intercept term must be provided. {params[1].shape[0]} intercepts provided instead!")
        return params


