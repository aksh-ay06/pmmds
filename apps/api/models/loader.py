"""Model loader with MLflow registry support.

Loads PySpark models from MLflow registry with fallback to dummy model.
Pre-warms SparkSession at startup for low-latency inference.
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Model(Protocol):
    """Protocol for model interface."""

    def predict(self, features: dict[str, Any]) -> float:
        """Generate fare prediction from features.

        Args:
            features: Dictionary of feature name -> value.

        Returns:
            Predicted fare amount.
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

    Returns fare predictions based on trip distance and duration.
    """

    _name: str = settings.model_name
    _version: str = "0.0.1-stub"

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def version(self) -> str:
        """Model version."""
        return self._version

    def predict(self, features: dict[str, Any]) -> float:
        """Generate dummy fare prediction.

        Uses trip distance and duration to create realistic fare estimates.

        Args:
            features: Input feature dictionary.

        Returns:
            Predicted fare amount.
        """
        seed_value = hash(frozenset(
            (k, v) for k, v in features.items() if not isinstance(v, (list, dict))
        )) % (2**32)
        rng = random.Random(seed_value)

        # Base fare ($3.00) + per-mile rate ($2.50) + per-minute rate ($0.50)
        trip_distance = features.get("trip_distance", 3.0)
        trip_duration = features.get("trip_duration_minutes", 15.0)

        base_fare = 3.00
        distance_fare = trip_distance * 2.50
        time_fare = trip_duration * 0.50

        # Rush hour surcharge
        is_rush_hour = features.get("is_rush_hour", 0)
        rush_surcharge = 2.50 if is_rush_hour else 0.0

        fare = base_fare + distance_fare + time_fare + rush_surcharge
        noise = rng.gauss(0, 1.0)
        fare = max(2.50, fare + noise)

        return round(fare, 2)


class MLflowModel:
    """Wrapper for MLflow pyfunc model (PySpark under the hood)."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        model_version: str,
    ) -> None:
        """Initialize MLflow model wrapper.

        Args:
            model: Loaded pyfunc model from MLflow.
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

    def predict(self, features: dict[str, Any]) -> float:
        """Generate fare prediction using MLflow pyfunc model.

        Args:
            features: Input feature dictionary.

        Returns:
            Predicted fare amount.
        """
        import pandas as pd

        # Convert to DataFrame (pyfunc expects DataFrame input)
        # Cast categoricals to string as training expects
        features_copy = features.copy()
        features_copy["RatecodeID"] = str(features_copy.get("RatecodeID", 1))
        features_copy["payment_type"] = str(features_copy.get("payment_type", 1))

        df = pd.DataFrame([features_copy])

        # Get prediction
        prediction = self._model.predict(df)
        fare = float(prediction[0])

        return round(max(2.50, fare), 2)


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

            # Load as pyfunc (handles Spark models transparently)
            loaded_model = mlflow.pyfunc.load_model(model_uri)

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
_model_loader_lock = threading.Lock()


def get_model_loader() -> ModelLoader:
    """Get singleton model loader instance.

    Returns:
        ModelLoader instance.
    """
    global _model_loader
    if _model_loader is None:
        with _model_loader_lock:
            if _model_loader is None:
                _model_loader = ModelLoader()
    return _model_loader


def reset_model_loader() -> None:
    """Reset model loader (for testing)."""
    global _model_loader
    _model_loader = None
