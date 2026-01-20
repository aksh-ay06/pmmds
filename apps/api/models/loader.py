"""Model loader stub.

Provides a dummy model for initial development.
Will be replaced with MLflow model loading in milestone 2.
"""

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from shared.utils import get_logger

logger = get_logger(__name__)


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
    """Stub model for development.

    Returns random predictions with realistic probability distribution.
    Simulates ~27% churn rate (typical for telco datasets).
    """

    _name: str = "dummy-churn-model"
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


class ModelLoader:
    """Manages model loading and caching.

    Currently loads DummyModel. Will integrate with MLflow registry.
    """

    def __init__(self) -> None:
        """Initialize model loader."""
        self._models: dict[str, Model] = {}
        logger.info("model_loader_initialized")

    def load(self, name: str = "default", version: str = "latest") -> Model:
        """Load a model by name and version.

        Args:
            name: Model name (ignored for stub).
            version: Model version (ignored for stub).

        Returns:
            Loaded model instance.
        """
        cache_key = f"{name}:{version}"

        if cache_key not in self._models:
            logger.info("loading_model", name=name, version=version)
            # TODO: Replace with MLflow model loading
            self._models[cache_key] = DummyModel()

        return self._models[cache_key]

    def get_current(self) -> Model:
        """Get the currently active model.

        Returns:
            Current production model.
        """
        return self.load()


@lru_cache
def get_model_loader() -> ModelLoader:
    """Get singleton model loader instance."""
    return ModelLoader()
