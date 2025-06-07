# pytorch_lightning_inference.py
"""
PyTorch Lightning LSTM инференс через MLflow Serving (без ONNX).
Использует PyTorch Lightning модель напрямую через MLflow registry.
"""

import json
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from warnings import warn

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import requests
import torch
import torch.nn as nn
from hydra import utils
from joblib import load as joblib_load
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

# Импорты из вашего проекта
from ...paths.paths import NAMES
from .load import (
    CLASS_NAME_TO_LABEL_MAP,
    CLASS_NAMES_ORDERED,
    LABEL_TO_CLASS_NAME_MAP,
    NUM_CLASSES,
)


class InferenceGaitSequenceDataset(Dataset):
    """Датасет для инференса LSTM модели."""

    def __init__(
        self, sequences: List[torch.Tensor], labels: List[int], file_ids: List[str]
    ):
        if not (len(sequences) == len(labels) == len(file_ids)):
            raise ValueError("Длины sequences, labels и file_ids должны совпадать")

        self.sequences = sequences
        self.labels = labels
        self.file_ids = file_ids

        seq_shape = sequences[0].shape if sequences else "(пусто)"
        print(
            f"Создан Inference датасет с {len(self.sequences)} последовательностями. Форма: {seq_shape}"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        return self.sequences[idx], self.labels[idx], self.file_ids[idx]


def load_sequences_for_inference(
    feature_dir: Path,
    names_structure: List[Dict[str, Any]],
    class_map: Dict[str, int],
    seq_length: int,
    stride: int,
    input_size_per_frame: int,
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Загружает .npy файлы и создает последовательности для инференса."""

    all_sequences, all_true_labels, all_file_identifiers = [], [], []
    print(f"\nЗагрузка последовательностей из {feature_dir}...")

    processed_files, skipped_files = 0, 0

    # Создаем карту filename -> label
    filename_to_label_map: Dict[str, int] = {}
    for class_info in names_structure:
        class_name = class_info.get("class")
        label = class_map.get(class_name)
        if label is None:
            continue
        for sample in class_info.get("samples", []):
            output_name = sample.get("out")
            if output_name:
                filename_to_label_map[output_name] = label

    print(f"Создана карта имя_файла -> метка для {len(filename_to_label_map)} файлов")

    if not feature_dir.is_dir():
        warn(f"Директория не найдена: {feature_dir}")
        return [], [], []

    # Загружаем .npy файлы
    for filename in sorted(feature_dir.glob("*.npy")):
        base_name = filename.stem
        true_label = filename_to_label_map.get(base_name)

        if true_label is None:
            warn(f"Не найдена метка для файла {base_name}")
            skipped_files += 1
            continue

        try:
            data = np.load(filename).astype(np.float32)

            if (
                data.ndim != 2
                or data.shape[0] < seq_length
                or data.shape[1] != input_size_per_frame
            ):
                raise ValueError(f"Некорректная форма данных: {data.shape}")

            # Создаем последовательности из файла
            sequences_from_file = []
            for i in range(0, data.shape[0] - seq_length + 1, stride):
                sequence = data[i : i + seq_length, :]
                sequences_from_file.append(torch.tensor(sequence, dtype=torch.float32))

            if sequences_from_file:
                num_seqs = len(sequences_from_file)
                all_sequences.extend(sequences_from_file)
                all_true_labels.extend([true_label] * num_seqs)
                all_file_identifiers.extend([base_name] * num_seqs)
                processed_files += 1
            else:
                warn(f"Не удалось создать последовательности из {filename}")
                skipped_files += 1

        except Exception as e:
            warn(f"Ошибка обработки {filename}: {e}")
            skipped_files += 1

    print(f"Создание последовательностей завершено.")
    print(f"Обработано файлов: {processed_files}")
    print(f"Пропущено файлов: {skipped_files}")
    print(f"Всего последовательностей: {len(all_sequences)}")

    return all_sequences, all_true_labels, all_file_identifiers


def start_mlflow_serving(cfg: DictConfig, model_uri: str) -> subprocess.Popen:
    """Запускает MLflow serving сервер для PyTorch Lightning модели."""

    print(f"🚀 Запуск MLflow serving сервера...")
    print(f"📦 Model URI: {model_uri}")
    print(f"🌐 Endpoint: http://{cfg.mlflow.serving.host}:{cfg.mlflow.serving.port}")

    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-h",
        cfg.mlflow.serving.host,
        "-p",
        str(cfg.mlflow.serving.port),
        "--workers",
        str(cfg.mlflow.serving.workers),
        "--no-conda",
    ]

    try:
        # Запуск в фоновом режиме
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Ждем запуска сервера
        print("⏳ Ожидание запуска сервера...")
        time.sleep(15)  # ✅ Увеличиваем время ожидания для PyTorch моделей

        # Проверяем готовность
        health_url = f"http://{cfg.mlflow.serving.host}:{cfg.mlflow.serving.port}/ping"
        for attempt in range(10):  # ✅ Больше попыток
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print("✅ MLflow serving сервер запущен успешно")
                    return process
            except requests.exceptions.RequestException:
                print(f"Попытка {attempt + 1}/10...")
                time.sleep(3)

        raise RuntimeError("Не удалось дождаться запуска MLflow сервера")

    except Exception as e:
        print(f"❌ Ошибка запуска MLflow serving: {e}")
        raise


def predict_via_mlflow_serving(
    sequences_batch: np.ndarray, endpoint_url: str, timeout: int = 30
) -> np.ndarray:
    """Выполняет предсказание через MLflow serving API для PyTorch Lightning модели."""

    # ✅ ФОРМАТ ДЛЯ PYTORCH LIGHTNING МОДЕЛИ
    sequences_batch_f32 = sequences_batch.astype(np.float32)

    data_dict = {
        "instances": sequences_batch_f32.tolist()  # PyTorch модели ожидают instances
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            endpoint_url, data=json.dumps(data_dict), headers=headers, timeout=timeout
        )

        if response.status_code != 200:
            print(f"❌ Status Code: {response.status_code}")
            print(f"❌ Response Text: {response.text}")
            print(f"❌ Request Data Shape: {sequences_batch.shape}")
            print(f"❌ Request Data Type: {sequences_batch_f32.dtype}")

        response.raise_for_status()

        predictions = response.json()

        # ✅ ОБРАБОТКА ОТВЕТА ОТ PYTORCH LIGHTNING МОДЕЛИ
        if "predictions" in predictions:
            result = np.array(predictions["predictions"])
        elif isinstance(predictions, list):
            result = np.array(predictions)
        else:
            result = np.array(predictions)

        print(f"✅ Получены предсказания. Форма: {result.shape}")
        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP ошибка: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"❌ Response content: {e.response.text}")
        raise RuntimeError(f"Ошибка HTTP запроса: {e}")
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        raise RuntimeError(f"Ошибка парсинга ответа: {e}")


def run_mlflow_inference_with_margin(
    dataloader: DataLoader, endpoint_url: str, batch_size: int, timeout: int = 30
) -> List[Dict[str, Any]]:
    """Выполняет инференс через MLflow serving с расчетом margin."""

    print("\n--- Запуск инференса через MLflow serving ---")

    # ✅ ТЕСТОВЫЙ ЗАПРОС
    print("🧪 Тестовый запрос с одним примером...")
    test_batch = next(iter(dataloader))
    test_sequences, _, _ = test_batch
    test_sample = test_sequences[:1].numpy()

    try:
        test_output = predict_via_mlflow_serving(test_sample, endpoint_url, timeout)
        print(f"✅ Тестовый запрос успешен. Выход: {test_output.shape}")
        print(f"📊 Пример предсказания: {test_output[0][:5]}")
        print(f"📊 Пример логитов (сумма): {test_output[0].sum():.4f}")
    except Exception as e:
        print(f"❌ Тестовый запрос неудачен: {e}")
        return []

    results: List[Dict[str, Any]] = []
    processed_sequences_count = 0

    for batch_idx, (sequences, true_labels_batch, file_ids_batch) in enumerate(
        dataloader
    ):
        batch_actual_size = sequences.size(0)
        sequences_np = sequences.numpy()

        try:
            # Предсказание через MLflow API
            outputs = predict_via_mlflow_serving(sequences_np, endpoint_url, timeout)

            # ✅ РАСЧЕТ ВЕРОЯТНОСТЕЙ И MARGIN (как в paste.txt)
            if outputs.ndim == 2:
                probs_batch = torch.softmax(torch.tensor(outputs), dim=1).numpy()
            else:
                probs_batch = outputs  # Если уже вероятности

            # Расчет margin (разность между топ-1 и топ-2)
            top_k_indices = np.argsort(probs_batch, axis=1)[:, -2:]  # Топ-2 индекса
            top_probs = probs_batch[
                np.arange(batch_actual_size), top_k_indices[:, -1]
            ]  # Топ-1
            second_probs = probs_batch[
                np.arange(batch_actual_size), top_k_indices[:, -2]
            ]  # Топ-2
            margins = top_probs - second_probs
            preds = top_k_indices[:, -1]  # Предсказания = топ-1 индексы

            # Собираем результаты
            true_labels_np = np.array(true_labels_batch)

            for i in range(batch_actual_size):
                results.append(
                    {
                        "predicted_label": preds[i],
                        "true_label": true_labels_np[i],
                        "file_id": file_ids_batch[i],
                        "margin": margins[i],
                        "top1_prob": top_probs[i],
                        "probabilities": probs_batch[i],
                    }
                )

            processed_sequences_count += batch_actual_size

            if (batch_idx + 1) % 10 == 0:
                print(f"Обработано батчей: {batch_idx + 1}/{len(dataloader)}")

        except Exception as e:
            warn(f"Ошибка в батче {batch_idx}: {e}")
            continue

    print(
        f"Инференс завершен. Обработано последовательностей: {processed_sequences_count}"
    )
    return results


def aggregate_and_report_margin_voting(
    all_sequence_results: List[Dict[str, Any]],
    label_to_name_map: Dict[int, str],
    num_classes: int,
    margin_threshold: float = 0.0,
    top_k: int = 3,
) -> Tuple[List[int], List[int], Dict[str, Dict]]:
    """Агрегирует результаты с margin sum voting как в paste.txt."""

    if not all_sequence_results:
        warn("Нет данных для агрегации")
        return [], [], {}

    print(
        f"\n--- Агрегация результатов с Margin Sum Voting (порог={margin_threshold}) ---"
    )

    # Фильтрация по margin
    filtered_results = [
        res for res in all_sequence_results if res["margin"] >= margin_threshold
    ]

    total_sequences = len(all_sequence_results)
    filtered_count = len(filtered_results)

    if total_sequences > 0:
        print(
            f"Отфильтровано последовательностей: {filtered_count}/{total_sequences} "
            f"({filtered_count/total_sequences*100:.1f}%)"
        )

    if filtered_count == 0:
        warn("После фильтрации не осталось последовательностей")
        return [], [], {}

    # Группировка по file_id
    file_data = defaultdict(
        lambda: {"preds_margins": [], "probs": [], "true_label": -1}
    )

    for res in filtered_results:
        file_id = res["file_id"]
        if file_data[file_id]["true_label"] == -1:
            file_data[file_id]["true_label"] = res["true_label"]

        file_data[file_id]["preds_margins"].append(
            (res["predicted_label"], res["margin"])
        )
        file_data[file_id]["probs"].append(res["probabilities"])

    # Агрегация с margin sum voting
    file_summaries: Dict[str, Dict] = {}
    final_true_labels: List[int] = []
    final_predicted_labels: List[int] = []
    correct_files_count = 0

    print("\n" + "=" * 50 + " Отчет по файлам " + "=" * 50)

    for file_id, data in sorted(file_data.items()):
        true_label = data["true_label"]
        true_name = label_to_name_map.get(true_label, f"Label_{true_label}")
        filtered_sequences_count = len(data["preds_margins"])

        # Margin sum voting
        margin_sums = np.zeros(num_classes, dtype=np.float64)
        for pred_label, margin_val in data["preds_margins"]:
            if 0 <= pred_label < num_classes:
                margin_sums[pred_label] += margin_val

        final_pred_label = np.argmax(margin_sums)
        final_pred_name = label_to_name_map.get(
            final_pred_label, f"Label_{final_pred_label}"
        )
        final_pred_score = margin_sums[final_pred_label]

        # Топ-K по средней вероятности
        avg_prob_dist = np.mean(data["probs"], axis=0)
        top_indices = np.argsort(avg_prob_dist)[-top_k:][::-1]

        # Проверка корректности
        is_correct = final_pred_label == true_label
        if is_correct:
            correct_files_count += 1

        status_str = "[Правильно]" if is_correct else "[Неправильно]"

        # Вывод отчета для файла
        print(f"\nФайл: {file_id}")
        print(f"  Истинный класс: {true_name}")
        print(f"  Предсказано (Margin Sum): {final_pred_name} {status_str}")
        print(f"  Итоговый счет: {final_pred_score:.4f}")
        print(f"  Топ-{top_k} классов (по средней вероятности):")
        for i, k_idx in enumerate(top_indices):
            k_name = label_to_name_map.get(k_idx, f"Label_{k_idx}")
            k_prob = avg_prob_dist[k_idx]
            print(f"    {i+1}. {k_name}: {k_prob:.4f}")
        print(f"  Учтено последовательностей: {filtered_sequences_count}")

        # Сохраняем данные
        final_true_labels.append(true_label)
        final_predicted_labels.append(final_pred_label)
        file_summaries[file_id] = {
            "true_label": true_label,
            "true_name": true_name,
            "predicted_label": final_pred_label,
            "predicted_name": final_pred_name,
            "predicted_score": float(final_pred_score),
            "is_correct": is_correct,
            "num_sequences_considered": filtered_sequences_count,
        }

    print("=" * 120)

    # Общая точность
    total_files = len(file_summaries)
    if total_files > 0:
        file_accuracy = correct_files_count / total_files
        print(
            f"\nИтоговая точность на уровне файлов: {file_accuracy:.4f} "
            f"({correct_files_count}/{total_files})"
        )

    return final_true_labels, final_predicted_labels, file_summaries


@hydra.main(
    config_path="../../../../configs/inference",
    config_name="mlflow_lstm",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """Главная функция инференса PyTorch Lightning модели через MLflow."""

    print("=== PyTorch Lightning LSTM инференс через MLflow Serving ===")
    print(f"Endpoint: http://{cfg.mlflow.serving.host}:{cfg.mlflow.serving.port}")
    print(f"Модель: {cfg.mlflow.model.model_uri}")
    print("=" * 50)

    # Получаем корневую директорию проекта
    project_root = Path(utils.get_original_cwd())

    # Настройка MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Tracking server

    serving_process = None

    try:
        # 1. Загрузка scaler
        print("\n--- Загрузка Scaler ---")
        scaler_path = project_root / cfg.data.scaler_path

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler не найден: {scaler_path}")

        scaler = joblib_load(scaler_path)
        print(f"✅ Scaler загружен: {scaler_path}")

        # 2. Загрузка данных
        print("\n--- Загрузка данных для инференса ---")
        feature_dir = project_root / cfg.data.input_dir

        sequences, true_labels, file_identifiers = load_sequences_for_inference(
            feature_dir=feature_dir,
            names_structure=NAMES,
            class_map=CLASS_NAME_TO_LABEL_MAP,
            seq_length=cfg.data.sequence_length,
            stride=cfg.data.stride,
            input_size_per_frame=cfg.data.input_size_per_frame,
        )

        if not sequences:
            print("❌ Нет данных для инференса")
            return

        # 3. Нормализация данных
        print("\n--- Нормализация данных ---")
        scaled_sequences = []
        for seq in sequences:
            seq_np = seq.numpy().astype(np.float32)
            seq_scaled = scaler.transform(seq_np).astype(np.float32)
            scaled_sequences.append(torch.tensor(seq_scaled, dtype=torch.float32))

        print(f"✅ Нормализовано {len(scaled_sequences)} последовательностей")

        # 4. Создание DataLoader
        print("\n--- Создание DataLoader ---")
        inference_dataset = InferenceGaitSequenceDataset(
            scaled_sequences, true_labels, file_identifiers
        )

        inference_loader = DataLoader(
            inference_dataset,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            num_workers=cfg.inference.num_workers,
            pin_memory=cfg.inference.pin_memory,
        )

        # 5. Запуск MLflow serving
        serving_process = start_mlflow_serving(cfg, cfg.mlflow.model.model_uri)

        # 6. Инференс через MLflow API
        endpoint_url = cfg.client.endpoint_url

        all_sequence_results = run_mlflow_inference_with_margin(
            dataloader=inference_loader,
            endpoint_url=endpoint_url,
            batch_size=cfg.client.max_batch_size,
            timeout=cfg.client.timeout_seconds,
        )

        # 7. Агрегация результатов
        if all_sequence_results:
            (
                final_true_labels,
                final_predicted_labels,
                file_summaries,
            ) = aggregate_and_report_margin_voting(
                all_sequence_results=all_sequence_results,
                label_to_name_map=LABEL_TO_CLASS_NAME_MAP,
                num_classes=NUM_CLASSES,
                margin_threshold=cfg.inference.aggregation.margin_threshold,
                top_k=cfg.inference.aggregation.top_k_classes,
            )

            # 8. Classification Report
            if final_true_labels:
                print(f"\n{'-'*20} Classification Report {'-'*20}")

                present_labels = sorted(
                    list(set(final_true_labels) | set(final_predicted_labels))
                )
                present_labels = [lbl for lbl in present_labels if lbl >= 0]

                if present_labels:
                    target_names = [
                        LABEL_TO_CLASS_NAME_MAP.get(lbl, f"Label_{lbl}")[:25]
                        for lbl in present_labels
                    ]

                    report_str = classification_report(
                        final_true_labels,
                        final_predicted_labels,
                        labels=present_labels,
                        target_names=target_names,
                        zero_division=0,
                        digits=3,
                    )
                    print(report_str)

                    experiment_name = "LSTM_Inference_Results"
                    try:
                        experiment = mlflow.get_experiment_by_name(experiment_name)
                        if experiment is None:
                            experiment_id = mlflow.create_experiment(experiment_name)
                        else:
                            experiment_id = experiment.experiment_id
                        mlflow.set_experiment(experiment_name)
                    except Exception as e:
                        print(
                            f"Предупреждение: не удалось создать эксперимент MLflow: {e}"
                        )
                        # Продолжаем без логирования
                        experiment_id = None

                    # ✅ ТЕПЕРЬ БЕЗОПАСНО ЗАПУСКАЕМ RUN
                    if experiment_id is not None:
                        with mlflow.start_run(
                            run_name="PyTorch_Lightning_LSTM_Inference"
                        ):
                            accuracy = sum(
                                1
                                for t, p in zip(
                                    final_true_labels, final_predicted_labels
                                )
                                if t == p
                            ) / len(final_true_labels)
                            mlflow.log_metric("file_level_accuracy", accuracy)
                            mlflow.log_metric(
                                "total_files_processed", len(file_summaries)
                            )
                            mlflow.log_metric(
                                "total_sequences_processed", len(all_sequence_results)
                            )

                            # Остальное логирование...
                            print(f"✅ Результаты логированы в MLflow")
                    else:
                        print(
                            "⚠️ Пропускаем логирование в MLflow из-за ошибки эксперимента"
                        )

                    # 9. Логирование результатов в MLflow
                    with mlflow.start_run(run_name="PyTorch_Lightning_LSTM_Inference"):
                        accuracy = sum(
                            1
                            for t, p in zip(final_true_labels, final_predicted_labels)
                            if t == p
                        ) / len(final_true_labels)
                        mlflow.log_metric("file_level_accuracy", accuracy)
                        mlflow.log_metric("total_files_processed", len(file_summaries))
                        mlflow.log_metric(
                            "total_sequences_processed", len(all_sequence_results)
                        )

                        # Логируем параметры инференса
                        mlflow.log_params(
                            {
                                "margin_threshold": cfg.inference.aggregation.margin_threshold,
                                "batch_size": cfg.inference.batch_size,
                                "aggregation_method": cfg.inference.aggregation.method,
                                "model_type": "PyTorch_Lightning",
                                "endpoint_url": endpoint_url,
                            }
                        )

                        print(f"✅ Результаты логированы в MLflow")

            # 10. ФИНАЛЬНЫЙ ВЫВОД: количество предсказанных людей
            unique_people = set(file_summaries.keys())
            total_people_count = len(unique_people)

            print(f"\n🎯 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
            print(f"📊 Всего предсказано людей: {total_people_count}")
            print(
                f"📈 Точность на уровне людей: {len([s for s in file_summaries.values() if s['is_correct']])}/{total_people_count}"
            )

            # Показать предсказанных людей по классам
            predicted_by_class = defaultdict(list)
            for file_id, summary in file_summaries.items():
                predicted_by_class[summary["predicted_name"]].append(file_id)

            print(f"\n👥 Распределение предсказаний по классам:")
            for class_name, people in sorted(predicted_by_class.items()):
                print(f"  {class_name}: {len(people)} человек")
        else:
            print("❌ Нет результатов инференса для обработки")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()
        return 1

    finally:
        # Останавливаем MLflow serving
        if serving_process:
            print("\n🛑 Остановка MLflow serving сервера...")
            serving_process.terminate()
            serving_process.wait()
            print("✅ MLflow сервер остановлен")

    print("\n--- PyTorch Lightning инференс завершен ---")
    return 0


if __name__ == "__main__":
    main()
