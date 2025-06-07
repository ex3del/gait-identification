import json
import os
from typing import TYPE_CHECKING

import lightgbm
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ...paths.paths import MODELS, NAMES, TRAIN

if TYPE_CHECKING:
    pass

WEIGHTS = MODELS / "LGBM" / "LGBM_weights"
print(MODELS)
# === Файлы ===
feature_dir = TRAIN.FEATURES
all_files = sorted([str(feature_dir / file) for file in os.listdir(feature_dir)])
names = {i["bag"]: n for n, i in enumerate(NAMES)}

# задайте диапазон параметров и параметры для перебора
PARAM_GRID = {
    "n_estimators": [400, 500, 600],
    "objective": ["multiclass"],
    "metric": ["multi_logloss"],
    "boosting_type": ["gbdt", "dart"],
    "subsample_for_bin": [200000, 300000, 400000],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "max_depth": [3, 5, 8],
    "min_data_in_leaf": [20, 60, 70],
    "feature_fraction": [0.4, 0.6, 0.8],
    "reg_alpha": [0.1, 0.2, 0.5],
    "reg_lambda": [0.1, 0.2, 0.5],
}
TRAIN_RATIO = 0.75  # Соотношение train/test


# === Загрузка данных ===
def load_and_split_data(file_paths, name_to_label_map):
    train_data, test_data = [], []
    train_labels, test_labels = [], []

    for file in file_paths:
        # Загрузка данных из файла
        arr = np.load(file)

        # Извлечение имени класса из пути файла
        class_name = file[31:-4]
        label = name_to_label_map[class_name]

        # Разделение данных без перемешивания
        split_idx = int(len(arr) * TRAIN_RATIO)

        # Добавление в тренировочные данные
        train_data.append(arr[:split_idx])
        train_labels.append(np.full(split_idx, label))

        # Добавление в тестовые данные
        test_data.append(arr[split_idx:])
        test_labels.append(np.full(len(arr) - split_idx, label))

    return (
        np.vstack(train_data),
        np.concatenate(train_labels),
        np.vstack(test_data),
        np.concatenate(test_labels),
    )


def main(
    scaler_path: str | os.PathLike = WEIGHTS / "LGBM_train_scaler.joblib",
    best_weights: str | os.PathLike = WEIGHTS / "best_model.txt",
    top_params: str | os.PathLike = WEIGHTS / "top_params.json",
    param_tuning=False,
):
    assert WEIGHTS.exists(), str(WEIGHTS)

    # === Загрузка и разделение данных ===
    X_train, y_train, X_test, y_test = load_and_split_data(all_files, names)

    # === Предобработка данных ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    if param_tuning:
        # Set up the k-fold cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=False)
        model = lightgbm.LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            boosting_type="gbdt",
            verbosity=10,
            n_jobs=-1,
        )
        random_search = RandomizedSearchCV(
            model,
            param_distributions=PARAM_GRID,
            n_iter=20,  # Количество итераций поиска
            cv=kfold,
            scoring="accuracy",
            verbose=10,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)

        print("Best Parameters: from Random Search", random_search.best_params_)
        best_params = random_search.best_params_
    else:
        best_params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "subsample_for_bin": 200000,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": 10,
            "min_data_in_leaf": 20,
            "feature_fraction": 1.0,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "n_estimators": 700,
        }

    # Inference with best found params
    best_model = lightgbm.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Сохранение модели
    best_model.booster_.save_model(str(best_weights))
    print(f"Model saved to {best_weights}")

    # Сохранение лучших параметров
    with open(top_params, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {top_params}")

    # Предсказание на тестовом наборе данных
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Точность модели: {:.2f}".format(accuracy))
    print("\n")
    print(
        classification_report(
            y_test, y_pred, labels=list(names.values()), target_names=list(names.keys())
        )
    )


if __name__ == "__main__":
    main(
        scaler_path=WEIGHTS / "LGBM_train_scaler.joblib",
        best_weights=WEIGHTS / "best_model.txt",
        top_params=WEIGHTS / "top_params.json",
        param_tuning=False,
    )
