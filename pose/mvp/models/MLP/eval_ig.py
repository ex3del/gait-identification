"""
Модуль для выполнения предсказаний (инференса) с использованием обученной MLP модели.

Скрипт загружает предварительно извлеченные признаки для оценочного набора данных,
применяет сохраненный StandardScaler, загружает обученную MLP модель и выполняет
предсказание классов для каждого окна/кадра.

Дополнительно, скрипт включает функционал для:
- Фильтрации предсказаний по порогу уверенности.
- Агрегации предсказаний на уровне файлов (например, через majority voting).
- Вывода сводной информации и отчета о классификации (если доступны истинные метки).
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import pandas as pd  # Для сохранения отчета
import torch
import torch.nn as nn

# import torch.optim as optim # Оптимизатор не нужен для инференса
from joblib import load  # Для загрузки scaler

# from sklearn.preprocessing import StandardScaler # StandardScaler загружается, не создается
from sklearn.metrics import f1_score  # Могут быть полезны для оценки итогового репорта
from sklearn.metrics import (  # confusion_matrix, ConfusionMatrixDisplay, # Можно добавить для визуализации
    accuracy_score,
    classification_report,
)
from torch.utils.data import DataLoader, Dataset

from ...feature_bake import main as feature_bake

try:
    from ...paths.paths import EVAL, MODELS, NAMES
except ImportError as e:
    print(f"Ошибка импорта путей: {e}. Убедитесь в корректности структуры проекта.")
    exit(1)

try:
    from .MLP import (  # GaitDataset,
        DROPOUT_RATE,
        HIDDEN_SIZE,
        HIDDEN_SIZE_2,
        USE_DEEP_MODEL,
        MLPClassifier,
        MLPClassifier_deep,
    )
except ImportError:
    # Если импорт не удался, копируем определения сюда (упрощенный вариант)
    print(
        "Предупреждение: Не удалось импортировать определения моделей/Dataset из .MLP. Используются локальные определения."
    )
if TYPE_CHECKING:
    pass


# --- Пути и Конфигурация ---
WEIGHTS = MODELS / "MLP" / "MLP_weights"
# Путь к сохраненному scaler'у
SCALER_PATH = WEIGHTS / "LSTM_train_scaler.joblib"
# Путь к обученным весам модели
# Используйте имя файла, сохраненное скриптом обучения
MODEL_WEIGHTS_PATH = WEIGHTS / "best_LSTM.pth"
# Директория с файлами признаков для предсказания
# Используем EVAL набор для оценки
EVAL_FEATURE_DIR = EVAL.FEATURES
# Путь для сохранения отчета
REPORT_CSV_PATH = WEIGHTS / "prediction_report_test.csv"

# Количество классов и NCOLS определим позже
NUM_CLASSES = -1
NCOLS = -1

# Настройки инференса
BATCH_SIZE = (
    3000000  # Размер батча для инференса (можно увеличить, т.к. нет обратного прохода)
)
# !!! ВАЖНО: Новый смысл CONFIDENCE_THRESHOLD !!!
# Теперь это порог для *margin* (разницы вероятностей), а не для top probability.
# Если хотите оставить фильтрацию по top probability, нужно вернуть старую логику
# или добавить отдельный порог. Сейчас будем фильтровать по margin >= CONFIDENCE_THRESHOLD.
CONFIDENCE_THRESHOLD = (
    0.1  # Минимальная уверенность для учета предсказания кадра при голосовании
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Генерация словарей меток (как в скрипте обучения) ---
try:
    CLASS_NAME_TO_LABEL_MAP: Dict[str, int] = {
        info["class"]: idx for idx, info in enumerate(NAMES) if "class" in info
    }
    if not CLASS_NAME_TO_LABEL_MAP:
        raise ValueError(
            "Структура NAMES не содержит записей с ключом 'class' или пуста."
        )
    LABEL_TO_CLASS_NAME_MAP: Dict[int, str] = {
        v: k for k, v in CLASS_NAME_TO_LABEL_MAP.items()
    }
    NUM_CLASSES = len(CLASS_NAME_TO_LABEL_MAP)  # Определяем количество классов
    print(f"Обнаружено классов: {NUM_CLASSES}")
    # pprint.pprint(CLASS_NAME_TO_LABEL_MAP)
except Exception as e:
    print(f"Ошибка при создании словаря меток из NAMES: {e}")
    exit(1)

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


class GaitDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# === Функция Загрузки Данных для Инференса ===
def load_inference_data(
    feature_dir: Path,
    output_to_label_map: Dict[str, int],  # Карта: имя_файла_без_расширения -> метка
) -> Tuple[np.ndarray | None, List[int] | None, List[str] | None]:
    """
    Загружает файлы признаков `.npy` из указанной директории для инференса/оценки.

    Итерирует по файлам в `feature_dir`. Для каждого файла `.npy`, чье имя
    (без расширения) присутствует в `output_to_label_map`, загружает данные.
    Возвращает объединенный массив данных, список истинных меток и список
    идентификаторов файлов для каждого сэмпла (окна/кадра).

    Args:
        feature_dir (Path): Директория с файлами признаков (`.npy`).
        output_to_label_map (Dict[str, int]): Словарь, отображающий базовое имя
                                              файла на его истинную метку класса.

    Returns:
        Tuple[np.ndarray | None, List[int] | None, List[str] | None]: Кортеж:
            - Массив NumPy с данными X (None, если нет данных).
            - Список истинных меток y_true (None, если нет данных).
            - Список идентификаторов файлов для каждого сэмпла (None, если нет данных).
    """
    X_list, y_true_list, file_identifiers_list = [], [], []
    print(f"Загрузка данных для инференса из: {feature_dir}")

    loaded_files_count = 0
    skipped_files_count = 0

    if not feature_dir.is_dir():
        warn(f"Директория {feature_dir} не найдена!")
        return None, None, None

    for filename in os.listdir(feature_dir):
        if filename.endswith(".npy"):
            base_name = filename[:-4]
            filepath = feature_dir / filename

            # Проверяем, есть ли метка для этого файла
            if base_name in output_to_label_map:
                label = output_to_label_map[base_name]
                try:
                    arr = np.load(filepath)
                    if arr.size > 0 and arr.ndim > 0:
                        X_list.append(arr)
                        y_true_list.append(np.full(len(arr), label, dtype=np.int32))
                        file_identifiers_list.append(
                            [base_name] * len(arr)
                        )  # Имя файла для каждого сэмпла
                        loaded_files_count += 1
                        # print(f"  Загружен файл: {filename} (метка: {label}), сэмплов: {len(arr)}")
                    else:
                        warn(f"  Пропущен пустой или некорректный файл: {filename}")
                        skipped_files_count += 1
                except Exception as e:
                    warn(f"  Ошибка загрузки файла {filename}: {e}")
                    skipped_files_count += 1
            # else:
            #     # Файл есть, но для него нет метки в карте - пропускаем молча или с предупреждением
            #     # warn(f"  Пропущен файл {filename}: отсутствует в output_to_label_map.")
            #     skipped_files_count += 1

    print(f"Загружено файлов: {loaded_files_count}")
    print(f"Пропущено файлов: {skipped_files_count}")

    if not X_list:
        warn("Не найдено или не загружено ни одного файла с признаками.")
        return None, None, None

    try:
        X_eval = np.concatenate(X_list, axis=0).astype(np.float32)
        y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)
        # Разворачиваем список списков идентификаторов
        file_identifiers = [
            item for sublist in file_identifiers_list for item in sublist
        ]

        print(f"Итоговая форма данных X_eval: {X_eval.shape}")
        print(f"Итоговая форма меток y_true: {y_true.shape}")
        print(f"Количество идентификаторов: {len(file_identifiers)}")

        return X_eval, y_true, file_identifiers

    except ValueError as e:
        print(f"\n!!! Ошибка при объединении массивов данных/меток: {e} !!!")
        print(
            "Возможно, массивы признаков из разных файлов имеют разное количество столбцов."
        )
        print("Формы некоторых загруженных массивов:")
        for i, arr in enumerate(X_list[:5]):
            print(f"  {i}: {arr.shape}")
        return None, None, None


# === Функция Инференса (МОДИФИЦИРОВАНА для расчета и фильтрации по MARGIN) ===
def run_inference(
    model: nn.Module,
    X_eval_scaled: np.ndarray,
    file_identifiers: List[str],
    true_labels: List[int],
    batch_size: int,
    n_cols: int,
    margin_threshold: float = 0.0,  # Переименован порог для ясности
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """
    Выполняет инференс, вычисляет margin и фильтрует результаты по margin_threshold.

    Args:
        model: Обученная модель PyTorch.
        X_eval_scaled: Масштабированные данные.
        file_identifiers: Идентификаторы файлов для каждого сэмпла.
        true_labels: Истинные метки для каждого сэмпла.
        batch_size: Размер батча.
        n_cols: Количество признаков.
        margin_threshold (float, optional): Минимальная разница (margin) между
                                             вероятностями top-1 и top-2 классов
                                             для учета сэмпла. Defaults to 0.0.

    Returns:
        Кортеж:
            - results: Список словарей для отфильтрованных сэмплов (включая 'margin').
            - filtered_true_labels: Истинные метки для отфильтрованных сэмплов.
            - filtered_preds: Предсказанные метки для отфильтрованных сэмплов.
    """
    global device
    print("\n--- Запуск инференса ---")
    print(f"Порог Margin (разница top1 - top2): {margin_threshold:.3f}")

    try:
        X_eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32).reshape(
            -1, 1, n_cols
        )
        eval_dataset = GaitDataset(X_eval_tensor)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Ошибка создания тензоров/DataLoader: {e}")
        return [], [], []

    model.eval()
    results = []
    filtered_true_labels = []
    filtered_preds = []
    frame_counter = 0

    with torch.no_grad():
        for inputs in eval_loader:
            batch_actual_size = inputs.size(0)
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # --- НОВОЕ: Расчет Margin ---
            # Получаем топ-2 вероятности и их индексы
            top_k_probs, top_k_indices = torch.topk(probs, 2, dim=1)

            top_probs = top_k_probs[:, 0]  # Вероятности top-1 класса
            second_probs = top_k_probs[:, 1]  # Вероятности top-2 класса
            margins = top_probs - second_probs  # Разница (margin)
            preds = top_k_indices[:, 0]  # Предсказанные метки (индексы top-1)
            # --- Конец нового блока ---

            # Обработка результатов батча
            for i in range(batch_actual_size):
                current_frame_idx = frame_counter + i
                margin_val = margins[i].item()

                # !!! Фильтрация по MARGIN !!!
                if margin_val >= margin_threshold:
                    predicted_label = preds[i].item()
                    top_prob_val = top_probs[i].item()  # Уверенность top-1
                    prob_distribution = probs[i].cpu().numpy()  # Полное распределение

                    # Убедимся, что индекс не выходит за пределы
                    if current_frame_idx < len(
                        file_identifiers
                    ) and current_frame_idx < len(true_labels):
                        file_id = file_identifiers[current_frame_idx]
                        true_label = true_labels[current_frame_idx]
                        results.append(
                            {
                                "file_identifier": file_id,
                                "predicted_class_label": predicted_label,
                                "true_class_label": true_label,
                                "confidence": top_prob_val,  # Сохраняем уверенность top-1
                                "margin": margin_val,  # Сохраняем margin
                                "probability_distribution": prob_distribution,
                            }
                        )
                        filtered_true_labels.append(true_label)
                        filtered_preds.append(predicted_label)
                    else:
                        warn(f"Индекс {current_frame_idx} вне диапазона.")

            frame_counter += batch_actual_size

    total_frames = len(X_eval_scaled)
    used_frames = len(results)
    if total_frames > 0:
        print(f"Обработано кадров/окон: {total_frames}")
        print(
            f"Учтено кадров/окон после фильтрации по margin: {used_frames} ({used_frames/total_frames*100:.1f}%)"
        )
    else:
        print("Нет данных для обработки.")
    print("-" * 20)

    return results, filtered_true_labels, filtered_preds


# === Функция Агрегации/Суммирования Результатов по Файлам ===
def summarize_predictions(
    results: List[Dict[str, Any]], label_to_name_map: Dict[int, str], num_classes: int
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Агрегирует предсказания, используя взвешенное голосование по margin.
    Класс файла определяется максимальной суммой margins для этого класса.
    Также вычисляет топ-3 класса по средней вероятности.

    Args:
        results: Список словарей результатов (включая 'margin').
        label_to_name_map: Словарь меток в имена.
        num_classes: Общее количество классов.

    Returns:
        Кортеж:
            - summaries: Словарь сводок по файлам.
            - correct_files_count: Количество правильно классифицированных файлов.
    """
    if not results:
        return {}, 0

    print(f"\n--- Агрегация предсказаний по файлам (Margin Sum Voting + Top 3) ---")

    file_predictions: Dict[str, Dict[str, Any]] = {}
    # Группировка и агрегация margins
    for result in results:
        file_id = result["file_identifier"]
        pred_label = result["predicted_class_label"]
        margin = result["margin"]
        prob_dist = result["probability_distribution"]
        true_label = result["true_class_label"]

        if file_id not in file_predictions:
            file_predictions[file_id] = {
                # Используем массив для хранения суммы маржи для каждого класса
                "margin_sums": np.zeros(num_classes, dtype=np.float64),
                # Используем список для сбора всех распределений вероятностей
                "prob_distributions": [],
                "frame_count": 0,
                "true_label": true_label,  # Предполагаем, что у всех кадров файла одна метка
            }

        file_predictions[file_id]["margin_sums"][pred_label] += margin
        file_predictions[file_id]["prob_distributions"].append(prob_dist)
        file_predictions[file_id]["frame_count"] += 1

    summaries: Dict[str, Dict[str, Any]] = {}
    correct_files_count = 0

    # Обработка каждого файла
    for file_id, data in file_predictions.items():
        if data["frame_count"] == 0:
            warn(f"Для файла '{file_id}' нет предсказаний после фильтрации.")
            continue

        # --- НОВОЕ: Определение победителя по сумме Margins ---
        margin_sums = data["margin_sums"]
        final_prediction_label = np.argmax(
            margin_sums
        )  # Класс с максимальной суммой маржи
        final_prediction_name = label_to_name_map.get(
            final_prediction_label, f"Label {final_prediction_label}"
        )
        max_margin_sum = margin_sums[
            final_prediction_label
        ]  # Значение максимальной суммы
        # --- Конец нового блока ---

        # --- Старая статистика (для информации) ---
        avg_prob_dist = (
            np.mean(data["prob_distributions"], axis=0)
            if data["prob_distributions"]
            else np.zeros(num_classes)
        )
        total_preds_for_file = data["frame_count"]

        # --- Определение Топ-3 по СРЕДНЕЙ ВЕРОЯТНОСТИ ---
        top_n = min(3, num_classes)
        top_n_indices = np.argsort(avg_prob_dist)[-top_n:][::-1]
        top_n_info = []
        for idx in top_n_indices:
            top_n_info.append(
                {
                    "label": idx,
                    "name": label_to_name_map.get(idx, f"Label {idx}"),
                    "average_probability": avg_prob_dist[idx],
                }
            )
        # --- Конец блока Топ-3 ---

        # Правильность предсказания файла (на основе margin sum)
        true_label_for_file = data["true_label"]
        is_correct = final_prediction_label == true_label_for_file
        if is_correct:
            correct_files_count += 1

        # Сохранение сводки
        summaries[file_id] = {
            "final_prediction_label": final_prediction_label,
            "final_prediction_name": final_prediction_name,
            "true_label": true_label_for_file,
            "true_name": label_to_name_map.get(
                true_label_for_file, f"Label {true_label_for_file}"
            ),
            "is_correct": is_correct,
            "total_margin_score": max_margin_sum,  # Сохраняем итоговый счет победителя
            "margin_sums_per_class": {  # Сохраняем все суммы для анализа
                label_to_name_map.get(idx, idx): score
                for idx, score in enumerate(margin_sums)
            },
            "average_probability_distribution": avg_prob_dist,
            "top_n_predictions_by_avg_prob": top_n_info,  # Уточнено имя
            "total_frames_considered": total_preds_for_file,
        }

    # Итоговая точность по файлам
    total_files_processed = len(summaries)
    if total_files_processed > 0:
        overall_file_accuracy = (correct_files_count / total_files_processed) * 100
        print(
            f"Правильно классифицировано файлов (по Margin Sum): {correct_files_count} из {total_files_processed} ({overall_file_accuracy:.2f}%)"
        )
    else:
        print("Не удалось агрегировать результаты ни для одного файла.")

    return summaries, correct_files_count


# === Основная Функция Запуска Инференса ===
def main(
    scaler_path: Path = SCALER_PATH,
    model_weights_path: Path = MODEL_WEIGHTS_PATH,
    eval_feature_dir: Path = EVAL_FEATURE_DIR,
    report_path: Path = REPORT_CSV_PATH,
    # Теперь это порог для MARGIN
    margin_threshold_val: float = CONFIDENCE_THRESHOLD,
    use_deep_model_flag: bool = USE_DEEP_MODEL,
):
    """Главная функция для запуска инференса MLP модели на оценочных данных."""
    global NCOLS, NUM_CLASSES

    print("--- Запуск пайплайна предсказания MLP (Margin Voting) ---")

    # --- 1. Проверка файлов ---
    if not scaler_path.is_file():
        print(f"Ошибка: Scaler не найден: {scaler_path}")
        return
    if not model_weights_path.is_file():
        print(f"Ошибка: Веса модели не найдены: {model_weights_path}")
        return
    if not eval_feature_dir.is_dir():
        print(f"Ошибка: Директория признаков не найдена: {eval_feature_dir}")
        return

    # --- 2. Загрузка Scaler ---
    try:
        scaler = load(scaler_path)
        print(f"Scaler загружен: {scaler_path}")
    except Exception as e:
        print(f"Ошибка загрузки scaler: {e}")
        return

    # --- 3. Загрузка Данных ---
    output_to_label_map: Dict[str, int] = {}
    for class_info in NAMES:
        class_name = class_info.get("class")
        label = CLASS_NAME_TO_LABEL_MAP.get(class_name)
        if label is None:
            continue
        for sample in class_info.get("samples", []):
            output_name = sample.get("out")
            if output_name:
                output_to_label_map[output_name] = label
    X_eval, y_true, file_identifiers = load_inference_data(
        eval_feature_dir, output_to_label_map
    )
    if X_eval is None:
        print("Ошибка загрузки данных.")
        return

    # --- 4. NCOLS и Масштабирование ---
    try:
        if X_eval.ndim > 1:
            NCOLS = X_eval.shape[1]
        elif X_eval.ndim == 1:
            NCOLS = 1
            X_eval = X_eval.reshape(-1, 1)
        else:
            raise ValueError("Не удалось определить NCOLS.")
        print(f"Определено признаков (NCOLS): {NCOLS}")
        print("Масштабирование данных...")
        X_eval_scaled = scaler.transform(X_eval)
    except Exception as e:
        print(f"Ошибка NCOLS/масштабирования: {e}")
        return

    # --- 5. Инициализация и Загрузка Модели ---
    print("Инициализация и загрузка модели...")
    try:
        if use_deep_model_flag:
            model = MLPClassifier_deep(
                NCOLS, HIDDEN_SIZE, HIDDEN_SIZE_2, NUM_CLASSES, DROPOUT_RATE
            ).to(device)
        else:
            model = MLPClassifier(NCOLS, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE).to(
                device
            )
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Веса модели загружены: {model_weights_path}")
    except Exception as e:
        print(f"Ошибка инициализации/загрузки модели: {e}")
        return

    # --- 6. Запуск Инференса ---
    # Передаем margin_threshold_val в run_inference
    frame_results, filtered_true_labels, filtered_preds = run_inference(
        model=model,
        X_eval_scaled=X_eval_scaled,
        file_identifiers=file_identifiers,
        true_labels=y_true,
        batch_size=BATCH_SIZE,
        n_cols=NCOLS,
        margin_threshold=margin_threshold_val,  # Используем порог для margin
    )
    if not frame_results:
        print("Инференс не дал результатов.")
        return

    # --- 7. Агрегация и Вывод Результатов ---
    file_summaries, correct_files_count = summarize_predictions(
        frame_results, LABEL_TO_CLASS_NAME_MAP, NUM_CLASSES
    )

    print(
        "\n"
        + "=" * 30
        + " Сводка Предсказаний по Файлам (Margin Sum Voting) "
        + "=" * 30
    )
    for file_id, summary in sorted(file_summaries.items()):
        correct_status = "Правильно" if summary["is_correct"] else "Неправильно"
        print(f"\nФайл: {file_id}")
        print(f"  Истинный класс: {summary['true_name']}")
        print(
            f"  Предсказано (Margin Sum): {summary['final_prediction_name']} ({correct_status})"
        )
        print(f"  Итоговый счет (Sum of Margins): {summary['total_margin_score']:.4f}")
        print(
            f"  Учтено кадров (с margin >= {margin_threshold_val:.3f}): {summary['total_frames_considered']}"
        )

        print("  Топ-3 предсказанных класса (по средней вероятности):")
        if "top_n_predictions_by_avg_prob" in summary:
            for i, top_pred in enumerate(summary["top_n_predictions_by_avg_prob"]):
                print(
                    f"    {i+1}. {top_pred['name']}: {top_pred['average_probability']:.4f}"
                )
        else:
            print("    Не удалось определить топ-3.")
        # Можно добавить вывод сумм маржи по классам для отладки:
        # print(f"  Суммы маржи по классам: {summary['margin_sums_per_class']}")
    print("=" * 80)

    # --- 8. Генерация и Сохранение Отчета (на кадрах, прошедших margin threshold) ---
    print(
        f"\n--- Отчет о классификации (на кадрах с margin >= {margin_threshold_val:.3f}) ---"
    )
    if filtered_true_labels and filtered_preds:
        present_labels = np.unique(
            np.concatenate((filtered_true_labels, filtered_preds))
        )
        present_names = [
            LABEL_TO_CLASS_NAME_MAP.get(i, f"L{i}") for i in present_labels
        ]
        try:
            report_dict = classification_report(
                filtered_true_labels,
                filtered_preds,
                labels=present_labels,
                target_names=present_names,
                zero_division=0,
                output_dict=True,
            )
            print(
                classification_report(
                    filtered_true_labels,
                    filtered_preds,
                    labels=present_labels,
                    target_names=present_names,
                    zero_division=0,
                )
            )
            try:
                report_df = pd.DataFrame(report_dict).transpose()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_df.to_csv(report_path, index=True)
                print(f"\nОтчет сохранен в {report_path}")
            except Exception as e:
                warn(f"Не удалось сохранить отчет CSV: {e}")
        except Exception as e:
            print(f"Не удалось сгенерировать отчет: {e}")
    else:
        print("Нет данных для отчета (возможно, все кадры отфильтрованы по margin).")

    print("\n--- Пайплайн предсказания MLP завершен ---")


# --- Точка входа ---
if __name__ == "__main__":
    feature_bake(
        eval=True,
        make_dir=True,
        parametrization=True,
        extractor_n_frames=True,
    )
    main(
        scaler_path=SCALER_PATH,
        model_weights_path=MODEL_WEIGHTS_PATH,
        eval_feature_dir=EVAL_FEATURE_DIR,
        report_path=REPORT_CSV_PATH,
        # Передаем порог для margin
        margin_threshold_val=0.1,  # Используем глобальное значение
        use_deep_model_flag=USE_DEEP_MODEL,
    )
