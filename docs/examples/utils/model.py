import jax.numpy as jnp
from numpy.typing import ArrayLike
import nemos as nmo
from nemos.base_class import DESIGN_INPUT_TYPE
from typing import Optional, Tuple, Union, Literal


class GLM(nmo.glm.GLM):

    def fit(
            self,
            X: DESIGN_INPUT_TYPE,
            y: ArrayLike,
            init_params: Optional[
            Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]
            ] = None
    ) -> nmo.glm.GLM:
        super().fit(jnp.expand_dims(X, axis=1), jnp.expand_dims(y, axis=1), init_params=init_params)
        return self

    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        return super().predict(jnp.expand_dims(X, axis=1))

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "pseudo-r2-McFadden",
    ) -> jnp.ndarray:
        return super().score(jnp.expand_dims(X, axis=1), jnp.expand_dims(y, axis=1), score_type=score_type)



