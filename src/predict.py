
"""
Predicting new, unlabelled data with the final (refit) pipeline.

Added in week 9 (option A). The new file goes through exactly the same steps as the training data,
 the same cleaning, the same feature/target split, except `drop_duplicate_rows`: every row of
the new file needs a prediction, so no row is ever dropped. Everything data-specific (file paths,
id column, threshold) comes from the config's `predict` section.
"""
import os
import pandas as pd
from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new_data(pipeline, config: dict) -> pd.DataFrame:
    """
    Loads `config["predict"]["path"]`, cleans it and splits off the features like the training
    data, and predicts it with the already-fitted `pipeline`.

    Returns one row per input row: the id, the predicted probability of class 1 and the 0/1
    prediction (probability >= `config["predict"]["threshold"]`, default 0.5).
    """
    predict_config = config["predict"]
    id_column = config["diagnostics"]["id_column"]
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])

    df_raw = load_data(predict_config["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])          # no drop_duplicate_rows here
    X_new, _, _ = split_features_target(df_clean, config["data"], mnar_sources)

    proba = pipeline.predict_proba(X_new)[:, 1]
    predictions = pd.DataFrame({
        id_column: df_clean[id_column].values,
        "probability": proba,
        "prediction": (proba >= predict_config.get("threshold", 0.5)).astype(int),
    })

    if len(predictions) != len(df_raw):
        raise ValueError(f"{len(df_raw)} rows in, but {len(predictions)} predictions out: "
                         "cleaning must never drop rows from the data to predict.")
    return predictions


def save_predictions(predictions: pd.DataFrame, path: str) -> str:
    """Writes the predictions to `path` (the folder is created if needed) and returns the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    predictions.to_csv(path, index=False)
    return path
