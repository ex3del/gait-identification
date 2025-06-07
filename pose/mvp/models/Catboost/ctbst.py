import json
import os
from typing import TYPE_CHECKING

import catboost
import numpy as np
import pandas as pd
import shap
from joblib import dump
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ...paths.paths import MODELS, NAMES, TRAIN

if TYPE_CHECKING:
    pass

WEIGHTS = MODELS / "Catboost" / "Catboost_weights"

# === Файлы ===
feature_dir = TRAIN.FEATURES
all_files = sorted([str(feature_dir / file) for file in os.listdir(feature_dir)])
names = {i["bag"]: n for n, i in enumerate(NAMES)}

# задать диапазон параметров и параметры для перебора
PARAM_GRID = {
    "iterations": [400, 600, 800],
    # "loss_function":['CrossEntropy', 'MultiClass'],
    # "eval_metric": ['CrossEntropy','MultiClass'],
    "grow_policy": ["Lossguide", "Depthwise"],
    "bootstrap_type": ["Poisson", "Bernoulli", "MVS"],
    "sampling_frequency": ["PerTree", "PerTreeLevel"],
    "min_data_in_leaf": [2, 4, 6],
    "subsample": [0.3, 0.5, 0.7],
    "leaf_estimation_method": ["Gradient", "Newton", "Exact"],
    "learning_rate": [0.01, 0.1],
    "depth": range(3, 11),
    "random_strength": [0.5, 0.3, 0.1],
    "reg_lambda": [1, 0.1, 0.5],
    "auto_class_weights": ["Balanced"],
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

    # === Важность признаков ===


def feature_importance_ctb(model, X, path):
    file_path = os.path.join(path, "feature_importance_ctb.csv")
    feature_importance = model.get_feature_importance()
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    feature_importance_df.to_csv(file_path, index=False)
    print(f"Feature importance saved to {path}")


def feature_importance_shap(model, X, path):
    file_path = os.path.join(path, "feature_importance_shap.csv")

    # Create an explainer for the model
    explainer = shap.Explainer(model)

    # Get SHAP values for the input data
    shap_values = explainer(X)
    print(f"SHAP values shape: {np.shape(shap_values.values)}")
    feature_importance = np.abs(shap_values.values).mean(axis=(0, 2))

    # Ensure the importance values are 1D
    if feature_importance.ndim > 1:
        feature_importance = feature_importance.flatten()

    # Create a list of feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    feature_importance_df.to_csv(file_path, index=False)
    print(f"Shap Feature importance saved to {file_path}")


def return_weights(exp):
    exp_list = exp.as_map()[1]
    exp_list = sorted(exp_list, key=lambda x: x[0])
    return [x[1] for x in exp_list]


def feature_importance_lime(model, X, X_sample, path):
    file_path = os.path.join(path, "feature_importance_lime.csv")

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X,
        mode="classification",
        feature_names=feature_names,
        discretize_continuous=True,
    )

    # Initialize a list to collect the weights for each sample
    weights_list = []

    print("Started iteration")
    for x in X:
        # Explain the instance
        exp = explainer.explain_instance(
            x, model.predict_proba, num_features=len(feature_names)
        )

        # Get the feature weights for this instance
        weights = return_weights(exp)
        weights_list.append(weights)

    # Convert the weights list to a numpy array
    weights_array = np.array(weights_list)

    # Calculate the mean absolute weight for each feature
    feature_importance = np.abs(weights_array).mean(axis=0)

    # Create a DataFrame with feature names and importance values
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )

    # Sort by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Save the feature importance to a CSV file
    feature_importance_df.to_csv(file_path, index=False)

    # Print confirmation message
    print(f"Feature importance saved to {file_path}")


def main(
    scaler_path: str | os.PathLike = WEIGHTS / "Catboost_train_scaler.joblib",
    best_weights: str | os.PathLike = WEIGHTS / "best_model.cbm",
    top_params: str | os.PathLike = WEIGHTS / "top_params.json",
    feature_imp_pth: str
    | os.PathLike = MODELS / "Catboost" / "feature_importance_analys",
    param_tuning=False,
    do_feature_analys=True,
):
    assert WEIGHTS.exists(), str(WEIGHTS)

    # === Загрузка и разделение данных ===
    X_train, y_train, X_test, y_test = load_and_split_data(all_files, names)
    print(np.shape(X_train))
    # === Предобработка данных ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    if param_tuning:
        kfold = StratifiedKFold(n_splits=5, shuffle=False)
        ct = catboost.CatBoostClassifier(
            task_type="GPU",
            devices="0",
            loss_function="MultiClass",
            eval_metric="MultiClass",
            logging_level="Silent",
        )
        # set up random search. Could be better with Optuna
        random_search = ct.randomized_search(
            PARAM_GRID,
            X=X_train,
            y=y_train,
            cv=kfold,
            n_iter=400,
            verbose=True,
        )
        print("Best Parameters:", random_search["params"])
        best_params = random_search["params"]

    else:
        best_params = {
            "random_strength": 0.1,
            "bootstrap_type": "Bernoulli",
            "leaf_estimation_method": "Gradient",
            "iterations": 600,
            "sampling_frequency": "PerTree",
            "auto_class_weights": "Balanced",
            "grow_policy": "Depthwise",
            "l2_leaf_reg": 1,
            "subsample": 0.5,
            "depth": 10,
            "min_data_in_leaf": 2,
            "learning_rate": 0.01,
        }

    # inference with best found params
    best_model = catboost.CatBoostClassifier(
        task_type="GPU",
        devices="0",
        loss_function="MultiClass",
        eval_metric="MultiClass",
        **best_params,
        verbose=0,
    )
    best_model.fit(X_train, y_train)

    # Сохранение модели
    best_model.save_model(str(best_weights))
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

    if do_feature_analys:
        print("\n \n I am starting feature analys \n \n")
        for path, X in [("train", X_train)]:
            feature_importance_ctb(
                model=best_model, X=X, path=os.path.join(feature_imp_pth, path)
            )
            feature_importance_shap(
                model=best_model, X=X, path=os.path.join(feature_imp_pth, path)
            )

            # Анализ с помощью lime пока не использовать , очень долго работает
            """
            feature_importance_lime(
                                model = best_model,
                                X = X,
                                X_sample = X,
                                path = os.path.join(feature_imp_pth, path)
                                )
            """


if __name__ == "__main__":
    main(
        scaler_path=WEIGHTS / "Catboost_train_scaler.joblib",
        best_weights=WEIGHTS / "best_model.cbm",
        top_params=WEIGHTS / "top_params.json",
        feature_imp_pth=MODELS / "Catboost" / "feature_importance_analys",
        param_tuning=False,
        do_feature_analys=True,
    )
