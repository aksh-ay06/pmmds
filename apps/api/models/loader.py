"""Model loader with MLflow registry support.

Loads models from MLflow registry with fallback to dummy model.
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Model(Protocol):
    """Protocol for model interface."""

    def predict(self, features: dict[str, Any]) -> tuple[int, float]:
        """Generate prediction from features.

        Args:
            features: Dictionary of feature name -> value.

        Returns:
            Tuple of (predicted_class, probability).
        """
        ...

    @property
    def name(self) -> str:
        """Model name."""
        ...

    @property
    def version(self) -> str:
        """Model version."""
        ...


@dataclass
class DummyModel:
    """Stub model for development/fallback.

    Returns random predictions with realistic probability distribution.
    Simulates ~27% churn rate (typical for telco datasets).
    """

    _name: str = settings.model_name
    _version: str = "0.0.1-stub"
    _base_churn_rate: float = 0.27

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def version(self) -> str:
        """Model version."""
        return self._version

    def predict(self, features: dict[str, Any]) -> tuple[int, float]:
        """Generate dummy prediction.

        Uses feature values to create deterministic-ish predictions
        for testing consistency, with some randomness.

        Args:
            features: Input feature dictionary.

        Returns:
            Tuple of (predicted_class, probability).
        """
        # Create a seed from features for semi-deterministic behavior
        seed_value = hash(frozenset(features.items())) % (2**32)
        rng = random.Random(seed_value)

        # Base probability influenced by some key features
        base_prob = self._base_churn_rate

        # Adjust based on tenure (lower tenure = higher churn risk)
        tenure = features.get("tenure", 12)
        if tenure < 6:
            base_prob += 0.15
        elif tenure > 48:
            base_prob -= 0.10

        # Adjust based on contract type
        contract = features.get("contract", "Month-to-month")
        if contract == "Month-to-month":
            base_prob += 0.10
        elif contract == "Two year":
            base_prob -= 0.15

        # Adjust based on monthly charges
        monthly_charges = features.get("monthly_charges", 50.0)
        if monthly_charges > 80:
            base_prob += 0.05

        # Add noise and clamp
        noise = rng.gauss(0, 0.05)
        probability = max(0.01, min(0.99, base_prob + noise))

        # Determine class with threshold
        prediction = 1 if probability > 0.5 else 0

        logger.debug(
            "dummy_prediction",
            prediction=prediction,
            probability=probability,
            tenure=tenure,
            contract=contract,
        )

        return prediction, round(probability, 4)


class MLflowModel:
    """Wrapper for MLflow sklearn model."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        model_version: str,
    ) -> None:
        """Initialize MLflow model wrapper.

        Args:
            model: Loaded sklearn pipeline from MLflow.
            model_name: Registered model name.
            model_version: Model version string.
        """
        self._model = model
        self._name = model_name
        self._version = model_version

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def version(self) -> str:
        """Model version."""
        return self._version

    def predict(self, features: dict[str, Any]) -> tuple[int, float]:
        """Generate prediction using MLflow model.

        Args:
            features: Input feature dictionary.

        Returns:
            Tuple of (predicted_class, probability).
        """
        import pandas as pd

        # Convert to DataFrame (sklearn pipeline expects DataFrame)
        df = pd.DataFrame([features])

        # Get prediction and probability
        prediction = int(self._model.predict(df)[0])
        probabilities = self._model.predict_proba(df)[0]
        probability = float(probabilities[1])  # Probability of class 1 (churn)

        return prediction, round(probability, 4)


class ModelLoader:
    """Manages model loading from MLflow registry with fallback."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str | None = None,
        model_alias: str | None = None,
        fallback_to_dummy: bool = True,
    ) -> None:
        """Initialize model loader.

        Args:
            tracking_uri: MLflow tracking server URI.
            model_name: Registered model name.
            model_alias: Model alias (e.g., 'production').
            fallback_to_dummy: Use dummy model if MLflow unavailable.
        """
        self._tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self._model_name = model_name or settings.model_name
        self._model_alias = model_alias or settings.model_alias
        self._fallback_to_dummy = fallback_to_dummy
        self._cached_model: Model | None = None
        self._model_load_time: float | None = None

        logger.info(
            "model_loader_initialized",
            tracking_uri=self._tracking_uri,
            model_name=self._model_name,
            model_alias=self._model_alias,
        )

    def _load_from_mlflow(self) -> Model | None:
        """Attempt to load model from MLflow registry.

        Returns:
            MLflowModel if successful, None otherwise.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri)

            # Load model using alias
            model_uri = f"models:/{self._model_name}@{self._model_alias}"
            logger.info("loading_model_from_mlflow", model_uri=model_uri)

            # Get model version info
            client = mlflow.MlflowClient()
            model_version = client.get_model_version_by_alias(
                name=self._model_name,
                alias=self._model_alias,
            )
            version_str = model_version.version

            # Load the model
            loaded_model = mlflow.sklearn.load_model(model_uri)

            logger.info(
                "model_loaded_from_mlflow",
                model_name=self._model_name,
                version=version_str,
                alias=self._model_alias,
            )

            return MLflowModel(
                model=loaded_model,
                model_name=self._model_name,
                model_version=version_str,
            )

        except Exception as e:
            logger.warning(
                "mlflow_model_load_failed",
                error=str(e),
                error_type=type(e).__name__,
                model_name=self._model_name,
                model_alias=self._model_alias,
            )
            return None

    def load(self, force_reload: bool = False) -> Model:
        """Load the model, using cache if available.

        Args:
            force_reload: Force reload from MLflow even if cached.

        Returns:
            Loaded model instance.
        """
        if self._cached_model is not None and not force_reload:
            return self._cached_model

        start_time = time.perf_counter()

        # Try MLflow first
        model = self._load_from_mlflow()

        # Fallback to dummy if needed
        if model is None:
            if self._fallback_to_dummy:
                logger.warning(
                    "falling_back_to_dummy_model",
                    reason="MLflow model not available",
                )
                model = DummyModel()
            else:
                raise RuntimeError(
                    f"Failed to load model '{self._model_name}' from MLflow "
                    f"and fallback is disabled"
                )

        self._cached_model = model
        self._model_load_time = time.perf_counter() - start_time

        logger.info(
            "model_ready",
            model_name=model.name,
            model_version=model.version,
            load_time_ms=round(self._model_load_time * 1000, 2),
        )

        return model

    def get_current(self) -> Model:
        """Get the currently loaded model.

        Returns:
            Current model (loads if not yet loaded).
        """
        return self.load()

    def reload(self) -> Model:
        """Force reload model from MLflow.

        Returns:
            Freshly loaded model.
        """
        return self.load(force_reload=True)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model metadata.
        """
        model = self.get_current()
        return {
            "name": model.name,
            "version": model.version,
            "tracking_uri": self._tracking_uri,
            "alias": self._model_alias,
            "load_time_ms": (
                round(self._model_load_time * 1000, 2)
                if self._model_load_time
                else None
            ),
        }


# Global model loader instance
_model_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    """Get singleton model loader instance.

    Returns:
        ModelLoader instance.
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def reset_model_loader() -> None:
    """Reset model loader (for testing)."""
    global _model_loader
    _model_loader = None
