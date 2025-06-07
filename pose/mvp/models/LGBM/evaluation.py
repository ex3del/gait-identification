import os
from collections import Counter
from typing import TYPE_CHECKING

import lightgbm
import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import classification_report

from ...paths.paths import EVAL, MODELS, NAMES

if TYPE_CHECKING:
    pass

from mvp.bag2mp4 import main as bag2mp4
from mvp.feature_bake import main as feature_bake
from mvp.get3d import main as get3d
from mvp.yolo17 import main as yolo17

WEIGHTS = MODELS / "LGBM" / "LGBM_weights"
names = {i["bag"]: n for n, i in enumerate(NAMES)}


def summarize_predictions(results):
    # Group predictions by file
    file_predictions = {}

    for result in results:
        file_name = result["file_name"]
        if file_name not in file_predictions:
            file_predictions[file_name] = {
                "predictions": [],
                "confidences": [],
                "probability_distributions": [],
            }

        file_predictions[file_name]["predictions"].append(result["predicted_class"])
        file_predictions[file_name]["confidences"].append(result["confidence"])
        file_predictions[file_name]["probability_distributions"].append(
            result["probability_distribution"]
        )

    # Summarize each file's predictions
    summaries = {}
    for file_name, data in file_predictions.items():
        # Get the most common prediction
        pred_counter = Counter(data["predictions"])
        most_common_pred = pred_counter.most_common(1)[0][0]

        # Calculate prediction statistics
        total_predictions = len(data["predictions"])
        prediction_counts = dict(pred_counter)
        prediction_percentages = {
            k: (v / total_predictions) * 100 for k, v in prediction_counts.items()
        }

        # Calculate average confidence
        avg_confidence = np.mean(data["confidences"])

        # Calculate average probability distribution
        avg_prob_dist = np.mean(data["probability_distributions"], axis=0)

        summaries[file_name] = {
            "most_common_prediction": most_common_pred,
            "prediction_counts": prediction_counts,
            "prediction_percentages": prediction_percentages,
            "average_confidence": avg_confidence,
            "average_probability_distribution": avg_prob_dist,
            "total_predictions": total_predictions,
        }

    return summaries


def load_evaluation_data(eval_dir):
    data = []
    file_names = []  # Store file names to keep track of predictions
    true_labels = []
    for file in sorted(os.listdir(eval_dir)):
        file_path = os.path.join(eval_dir, file)
        arr = np.load(file_path)
        data.append(arr)
        # Repeat filename for each row
        file_names.extend([file] * arr.shape[0])
        class_name = file[:-4]  # Удаляем расширение .npy
        true_labels.extend([names[class_name]] * arr.shape[0])
    return np.vstack(data), file_names, true_labels


def lgbm_inference(best_weights, eval_dir, scaler):
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess evaluation data
    X_eval, file_names, true_labels = load_evaluation_data(eval_dir)
    X_eval = scaler.transform(X_eval)

    # Load LightGBM model
    model = lightgbm.Booster(model_file=str(best_weights))

    # Perform inference
    probs = model.predict(X_eval, raw_score=False)
    preds = np.argmax(probs, axis=1)

    # Create results dictionary
    results = []
    for idx, (pred, prob, file_name) in enumerate(zip(preds, probs, file_names)):
        result = {
            "file_name": file_name,
            "predicted_class": int(pred),
            "confidence": float(prob[pred]),
            "probability_distribution": prob.tolist(),
        }
        results.append(result)

    return results, true_labels, preds


def lgbm_main(
    scaler_path: str | os.PathLike = WEIGHTS / "LGBM_train_scaler.joblib",
    best_weights: str | os.PathLike = WEIGHTS / "best_model.txt",
):
    # Load the saved scaler
    scaler = load(scaler_path)
    eval_dir = EVAL.FEATURES
    name = {value: key for key, value in names.items()}

    results, true_labels, all_preds = lgbm_inference(best_weights, eval_dir, scaler)
    summaries = summarize_predictions(results)

    # Print summarized results
    print("\n=== Prediction Summaries ===")
    for file_name, summary in summaries.items():
        print(f"\nFile: {file_name}")
        print(f"Most Common Prediction: {name[summary['most_common_prediction']]}")
        print(f"Confidence: {summary['average_confidence']:.4f}")
        print(f"Total Predictions: {summary['total_predictions']}")

        print("\nPrediction Distribution:")
        for class_id, percentage in summary["prediction_percentages"].items():
            print(f"{name[class_id]}: {percentage:.1f}%")

        print("\nTop 3 most likely classes (average probabilities):")
        avg_probs = summary["average_probability_distribution"]
        top3_indices = np.argsort(avg_probs)[-3:][::-1]
        for i, idx in enumerate(top3_indices):
            print(f"{i+1}. {name[idx]}: {avg_probs[idx]:.4f}")

    # Generate and save classification report
    report = classification_report(
        true_labels,
        all_preds,
        labels=list(name.keys()),
        target_names=list(name.values()),
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()
    report_csv_path = WEIGHTS / "lgbm_classification_report.csv"
    report_df.to_csv(report_csv_path, index=True)
    print(f"\nClassification report saved to {report_csv_path}")

    print(
        "\n\n===============================CLASSIFICATION REPORT==================================\n\n"
    )
    print(
        classification_report(
            true_labels,
            all_preds,
            labels=list(name.keys()),
            target_names=list(name.values()),
            zero_division=0,
        )
    )


if __name__ == "__main__":
    # Prepare data for evaluation
    bag2mp4(eval=True)
    yolo17(eval=True)
    get3d(eval=True)
    feature_bake(eval=True)
    lgbm_main(
        scaler_path=WEIGHTS / "LGBM_train_scaler.joblib",
        best_weights=WEIGHTS / "best_model.txt",
    )
