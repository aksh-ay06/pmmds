"""Feature preprocessing pipeline using PySpark ML."""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)

from shared.data.dataset import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from shared.utils import get_logger

logger = get_logger(__name__)

# Categorical features that are string type (need StringIndexer)
STRING_CATEGORICAL_FEATURES = ["pickup_borough", "dropoff_borough"]
# Categorical features that are already numeric (just need OneHotEncoder via indexer)
NUMERIC_CATEGORICAL_FEATURES = ["RatecodeID", "payment_type"]


def create_spark_preprocessing_pipeline() -> Pipeline:
    """Create PySpark ML preprocessing pipeline.

    Stages:
    1. StringIndexer for string categorical features
    2. StringIndexer for numeric categorical features (treats as categorical)
    3. OneHotEncoder for all indexed features
    4. VectorAssembler to combine all features

    Returns:
        PySpark ML Pipeline for preprocessing.
    """
    stages = []

    # Index string categorical features
    indexed_cols = []
    for col in STRING_CATEGORICAL_FEATURES:
        indexer = StringIndexer(
            inputCol=col,
            outputCol=f"{col}_indexed",
            handleInvalid="keep",
        )
        stages.append(indexer)
        indexed_cols.append(f"{col}_indexed")

    # Index numeric categorical features (cast to string first handled in data prep)
    for col in NUMERIC_CATEGORICAL_FEATURES:
        indexer = StringIndexer(
            inputCol=col,
            outputCol=f"{col}_indexed",
            handleInvalid="keep",
        )
        stages.append(indexer)
        indexed_cols.append(f"{col}_indexed")

    # OneHotEncoder for all indexed columns
    encoded_cols = []
    for col in indexed_cols:
        encoded_col = col.replace("_indexed", "_encoded")
        encoder = OneHotEncoder(
            inputCol=col,
            outputCol=encoded_col,
            handleInvalid="keep",
        )
        stages.append(encoder)
        encoded_cols.append(encoded_col)

    # Assemble all features into a single vector
    assembler_inputs = NUMERIC_FEATURES + BINARY_FEATURES + encoded_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)

    logger.info(
        "spark_preprocessing_pipeline_created",
        numeric_features=len(NUMERIC_FEATURES),
        binary_features=len(BINARY_FEATURES),
        categorical_features=len(CATEGORICAL_FEATURES),
        total_stages=len(stages),
    )

    return pipeline
