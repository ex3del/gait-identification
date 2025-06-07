import json
import os
from typing import TYPE_CHECKING

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ...paths.paths import MODELS, NAMES, TRAIN

if TYPE_CHECKING:
    pass

# Use the same weights path structure
WEIGHTS = MODELS / "Xgboost" / "Xgboost_weights"

# === Files ===
feature_dir = TRAIN.FEATURES
all_files = sorted([str(feature_dir / file) for file in os.listdir(feature_dir)])
names = {i["bag"]: n for n, i in enumerate(NAMES)}

# Define parameter grid for XGBoost
PARAM_GRID = {
    "n_estimators": [400, 600, 800],
    "max_depth": list(range(3, 11)),
    "learning_rate": [0.01, 0.1],
    "subsample": [0.3, 0.5, 0.7],
    "min_child_weight": [2, 4, 6],
    "reg_lambda": [1, 0.1, 0.5],
    "tree_method": ["gpu_hist"],
}
TRAIN_RATIO = 0.75  # Train/test split ratio


# === Data Loading Function ===
def load_and_split_data(file_paths, name_to_label_map):
    train_data, test_data = [], []
    train_labels, test_labels = [], []

    for file in file_paths:
        # Load data from file
        arr = np.load(file)

        # Extract class name from file path
        class_name = file[31:-4]
        label = name_to_label_map[class_name]

        # Split data without shuffling
        split_idx = int(len(arr) * TRAIN_RATIO)

        # Append training data
        train_data.append(arr[:split_idx])
        train_labels.append(np.full(split_idx, label))

        # Append testing data
        test_data.append(arr[split_idx:])
        test_labels.append(np.full(len(arr) - split_idx, label))

    return (
        np.vstack(train_data),
        np.concatenate(train_labels),
        np.vstack(test_data),
        np.concatenate(test_labels),
    )


def main(
    scaler_path: str | os.PathLike = WEIGHTS / "Xgboost_train_scaler.joblib",
    best_weights: str | os.PathLike = WEIGHTS / "xgb_reg.pkl",
    top_params: str | os.PathLike = WEIGHTS / "top_params.json",
    param_tuning: bool = False,
):
    assert WEIGHTS.exists(), str(WEIGHTS)

    # === Load and split data ===
    X_train, y_train, X_test, y_test = load_and_split_data(all_files, names)

    # === Data Preprocessing ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    if param_tuning:
        # Set up k-fold cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=False)
        xgb_clf = XGBClassifier(
            objective="multi:softmax",
            num_class=len(names),
            tree_method="gpu_hist",
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        # Perform randomized search over the parameter grid
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=PARAM_GRID,
            n_iter=400,
            cv=kfold,
            verbose=1,
            n_jobs=-1,
            scoring="accuracy",
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        print("Best Parameters:", random_search.best_params_)
        best_params = random_search.best_params_
    else:
        best_params = {
            "n_estimators": 600,
            "max_depth": 10,
            "learning_rate": 0.01,
            "subsample": 0.5,
            "min_child_weight": 2,
            "reg_lambda": 1,
            "tree_method": "gpu_hist",
        }

    # === Train the best model ===
    best_model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(names),
        use_label_encoder=False,
        eval_metric="mlogloss",
        **best_params,
    )
    best_model.fit(X_train, y_train)

    # Save the model
    best_model.save_model(str(best_weights))
    print(f"Model saved to {best_weights}")

    # Save the best parameters
    with open(top_params, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {top_params}")

    # === Evaluate the model ===
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy: {:.2f}".format(accuracy))
    print("\n")
    print(
        classification_report(
            y_test, y_pred, labels=list(names.values()), target_names=list(names.keys())
        )
    )


if __name__ == "__main__":
    main(
        scaler_path=WEIGHTS / "Xgboost_train_scaler.joblib",
        best_weights=WEIGHTS / "xgb_reg.pkl",
        top_params=WEIGHTS / "top_params.json",
        param_tuning=True,
    )
