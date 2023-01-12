__author__ = "Jarne Verhaeghe, Jeroen Van Der Donckt"

from typing import Any

from .shap_explainer import (
    CatboostExplainer,
    DeepLearningExplainer,
    EnsembleExplainer,
    LGBMExplainer,
    LinearExplainer,
    ShapExplainer,
    XGBoostExplainer,
)


class ShapExplainerFactory:
    """Factory class for creating the appropriate ShapExplainer."""

    _explainer_models = [
        CatboostExplainer,
        LGBMExplainer,
        XGBoostExplainer,
        EnsembleExplainer,
        LinearExplainer,
        DeepLearningExplainer,
    ]

    @classmethod
    def get_explainer(cls, model: Any) -> ShapExplainer:
        """Get the shap explainer for the given model.

        Parameters
        ----------
        model: Any
            The model to get the shap explainer for.

        Returns
        -------
        ShapExplainer
            The shap explainer for the given model.

        """
        for explainer_class in cls._explainer_models:
            try:  # To avoid errors when the library is not installed
                if explainer_class.supports_model(model):
                    return explainer_class(model)
            except Exception:
                pass
        raise ValueError(f"Given model ({model}) is not yet supported by our explainer models")
