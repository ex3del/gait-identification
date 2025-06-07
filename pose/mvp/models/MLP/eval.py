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
import pprint
from typing import TYPE_CHECKING, Dict, List, Tuple, Any
from warnings import warn
from pathlib import Path
from collections import Counter

from joblib import load  # Для загрузки scaler
import numpy as np
import torch
import torch.nn as nn

# import torch.optim as optim # Оптимизатор не нужен для инференса
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# from sklearn.preprocessing import StandardScaler # StandardScaler загружается, не создается
from sklearn.metrics import (
    accuracy_score,
    f1_score,  # Могут быть полезны для оценки итогового репорта
    classification_report,
    # confusion_matrix, ConfusionMatrixDisplay, # Можно добавить для визуализации
)
import pandas as pd  # Для сохранения отчета


from ...feature_bake import main as feature_bake

try:
    from ...paths.paths import TRAIN, EVAL, NAMES, MODELS
except ImportError as e:
    print(f"Ошибка импорта путей: {e}. Убедитесь в корректности структуры проекта.")
    exit(1)

try:
    from .MLP import (
        MLPClassifier,
        MLPClassifier_deep,
        # GaitDataset,
        USE_DEEP_MODEL,
        HIDDEN_SIZE,
        HIDDEN_SIZE_2,
        DROPOUT_RATE,
        SCALER,
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
CONFIDENCE_THRESHOLD = (
    0.5  # Минимальная уверенность для учета предсказания кадра при голосовании
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


# === Функция Инференса ===
def run_inference(
    model: nn.Module,
    X_eval_scaled: np.ndarray,
    file_identifiers: List[str],
    true_labels: List[int],  # Используем как ground truth если нужно
    batch_size: int,
    n_cols: int,  # Количество признаков
    confidence_threshold: float = 0.0,  # Порог для фильтрации
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """
    Выполняет инференс на предоставленных данных с использованием обученной модели.

    Преобразует данные в тензоры, создает DataLoader, выполняет предсказание,
    фильтрует результаты по порогу уверенности и возвращает структурированный
    список результатов.

    Args:
        model (nn.Module): Обученная модель PyTorch.
        X_eval_scaled (np.ndarray): Массив данных (уже масштабированных!).
        file_identifiers (List[str]): Список идентификаторов файлов для каждого сэмпла в X_eval_scaled.
        true_labels (List[int]): Список истинных меток для каждого сэмпла (для отчетности).
        batch_size (int): Размер батча для DataLoader.
        n_cols (int): Количество признаков (столбцов) в данных X.
        confidence_threshold (float, optional): Минимальная уверенность предсказания
                                                (от 0.0 до 1.0) для его учета. Defaults to 0.0.

    Returns:
        Tuple[List[Dict[str, Any]], List[int], List[int]]: Кортеж:
            - results: Список словарей, каждый описывает результат для одного
                       *отфильтрованного* сэмпла/кадра/окна.
            - filtered_true_labels: Список истинных меток *только* для отфильтрованных сэмплов.
            - filtered_preds: Список предсказанных меток *только* для отфильтрованных сэмплов.
    """
    global device  # Используем глобальное устройство

    print("\n--- Запуск инференса ---")
    print(f"Порог уверенности: {confidence_threshold*100:.1f}%")

    # Преобразование в тензор и Dataset/DataLoader
    try:
        # Reshape как при обучении
        X_eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32).reshape(
            -1, 1, n_cols
        )
        # Dataset для инференса содержит только X
        eval_dataset = GaitDataset(X_eval_tensor)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Ошибка при создании тензоров/DataLoader'а для инференса: {e}")
        return [], [], []

    model.eval()  # Переводим модель в режим оценки
    results = []
    filtered_true_labels = []
    filtered_preds = []

    frame_counter = 0  # Индекс для доступа к file_identifiers и true_labels
    with torch.no_grad():
        for inputs in eval_loader:
            batch_actual_size = inputs.size(
                0
            )  # Фактический размер батча (может быть меньше на последнем)
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Вероятности для каждого класса
            confidences, preds = torch.max(
                probs, dim=1
            )  # Макс. вероятность (уверенность) и предск. класс

            # Обработка результатов батча
            for i in range(batch_actual_size):
                current_frame_idx = frame_counter + i
                confidence = confidences[i].item()
                predicted_label = preds[i].item()

                # Фильтрация по порогу уверенности
                if confidence >= confidence_threshold:
                    prob_distribution = probs[i].cpu().numpy()
                    file_id = file_identifiers[current_frame_idx]
                    true_label = true_labels[current_frame_idx]

                    results.append(
                        {
                            "file_identifier": file_id,  # Используем более общее имя
                            "predicted_class_label": predicted_label,
                            "true_class_label": true_label,
                            "confidence": confidence,
                            "probability_distribution": prob_distribution,
                        }
                    )
                    filtered_true_labels.append(true_label)
                    filtered_preds.append(predicted_label)

            frame_counter += batch_actual_size

    total_frames = len(X_eval_scaled)
    used_frames = len(results)
    print(f"Обработано кадров/окон: {total_frames}")
    print(
        f"Учтено кадров/окон после фильтрации: {used_frames} ({used_frames/total_frames*100:.1f}%)"
    )
    print("-" * 20)

    return results, filtered_true_labels, filtered_preds


def summarize_predictions(
    results: List[Dict[str, Any]],
    label_to_name_map: Dict[int, str],
    num_classes: int,  # Добавлено для размера avg_prob_dist
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Агрегирует предсказания на уровне кадров/окон в одно предсказание на файл (majority voting)
    и определяет топ-3 класса по средней вероятности.

    Args:
        results: Список словарей результатов для каждого отфильтрованного кадра/окна.
        label_to_name_map: Словарь для преобразования меток в имена.
        num_classes: Общее количество классов.

    Returns:
        Кортеж:
            - summaries: Словарь сводок по файлам, включая топ-3 предсказания.
            - correct_files_count: Количество правильно классифицированных файлов.
    """
    if not results:
        return {}, 0

    print("\n--- Агрегация предсказаний по файлам (Majority Voting + Top 3) ---")

    file_predictions: Dict[str, Dict[str, List]] = {}
    for result in results:
        file_id = result["file_identifier"]
        if file_id not in file_predictions:
            file_predictions[file_id] = {
                "predicted_labels": [],
                "confidences": [],
                "prob_distributions": [],
                "true_label": result["true_class_label"],
            }
        file_predictions[file_id]["predicted_labels"].append(
            result["predicted_class_label"]
        )
        file_predictions[file_id]["confidences"].append(result["confidence"])
        file_predictions[file_id]["prob_distributions"].append(
            result["probability_distribution"]
        )

    summaries: Dict[str, Dict[str, Any]] = {}
    correct_files_count = 0

    for file_id, data in file_predictions.items():
        if not data["predicted_labels"]:
            warn(f"Для файла '{file_id}' нет предсказаний после фильтрации.")
            continue

        # --- Majority Voting ---
        pred_counter = Counter(data["predicted_labels"])
        max_votes = pred_counter.most_common(1)[0][1]
        most_common_preds = [
            label for label, count in pred_counter.items() if count == max_votes
        ]
        final_prediction_label = most_common_preds[0]
        final_prediction_name = label_to_name_map.get(
            final_prediction_label, f"Label {final_prediction_label}"
        )

        # --- Статистика ---
        total_preds_for_file = len(data["predicted_labels"])
        avg_confidence = np.mean(data["confidences"]) if data["confidences"] else 0.0
        avg_prob_dist = (
            np.mean(data["prob_distributions"], axis=0)
            if data["prob_distributions"]
            # Убедимся, что массив имеет правильный размер, даже если data['prob_distributions'] пуст
            else np.zeros(num_classes)
        )

        # --- Определение Топ-3 классов ---
        top_n = min(3, num_classes)  # Берем топ 3 или меньше, если классов меньше
        # Индексы (метки) топ-N классов по средней вероятности (от большей к меньшей)
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

        # Правильность предсказания файла
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
            "vote_counts": dict(pred_counter),
            "vote_percentages": {
                label_to_name_map.get(k, k): (v / total_preds_for_file) * 100
                for k, v in pred_counter.items()
            },
            "average_confidence_for_file": avg_confidence,
            "average_probability_distribution": avg_prob_dist,  # Оставляем для возможного использования
            "top_n_predictions": top_n_info,  # Добавляем топ-N информацию
            "total_votes_for_file": total_preds_for_file,
        }

    # Итоговая точность по файлам
    total_files_processed = len(summaries)
    if total_files_processed > 0:
        overall_file_accuracy = (correct_files_count / total_files_processed) * 100
        print(
            f"Правильно классифицировано файлов (по majority vote): {correct_files_count} из {total_files_processed} ({overall_file_accuracy:.2f}%)"
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
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    use_deep_model_flag: bool = USE_DEEP_MODEL,
):
    """Главная функция для запуска инференса MLP модели на оценочных данных."""
    global NCOLS, NUM_CLASSES

    print("--- Запуск пайплайна предсказания MLP ---")

    # --- 1. Проверка наличия файлов ---
    if not scaler_path.is_file():
        print(f"Ошибка: Файл скейлера не найден: {scaler_path}")
        return
    if not model_weights_path.is_file():
        print(f"Ошибка: Файл с весами модели не найден: {model_weights_path}")
        return
    if not eval_feature_dir.is_dir():
        print(f"Ошибка: Директория с признаками не найдена: {eval_feature_dir}")
        return

    # --- 2. Загрузка Scaler ---
    try:
        scaler = load(scaler_path)
        print(f"StandardScaler загружен из {scaler_path}")
    except Exception as e:
        print(f"Ошибка загрузки скейлера: {e}")
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
        print("Ошибка: Не удалось загрузить данные для инференса.")
        return

    # --- 4. Определение NCOLS и Масштабирование ---
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
        print(f"Ошибка определения NCOLS или масштабирования: {e}")
        return

    # --- 5. Инициализация и Загрузка Модели ---
    print("Инициализация и загрузка модели...")
    try:
        # Динамическая инициализация модели
        if use_deep_model_flag:
            print("Используется MLPClassifier_deep")
            model = MLPClassifier_deep(
                NCOLS, HIDDEN_SIZE, HIDDEN_SIZE_2, NUM_CLASSES, DROPOUT_RATE
            ).to(device)
        else:
            print("Используется MLPClassifier")
            model = MLPClassifier(NCOLS, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE).to(
                device
            )

        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Веса модели загружены из {model_weights_path}")
    except FileNotFoundError:
        print(f"Ошибка: Файл весов не найден: {model_weights_path}")
        return
    except Exception as e:
        print(f"Ошибка инициализации/загрузки модели: {e}")
        print(f"  Параметры: NCOLS={NCOLS}, NUM_CLASSES={NUM_CLASSES}")
        return

    # --- 6. Запуск Инференса ---
    frame_results, filtered_true_labels, filtered_preds = run_inference(
        model=model,
        X_eval_scaled=X_eval_scaled,
        file_identifiers=file_identifiers,
        true_labels=y_true,
        batch_size=BATCH_SIZE,
        n_cols=NCOLS,
        confidence_threshold=confidence_threshold,
    )
    if not frame_results:
        print("Инференс не дал результатов.")
        return

    # --- 7. Агрегация и Вывод Результатов ---
    file_summaries, correct_files_count = summarize_predictions(
        frame_results, LABEL_TO_CLASS_NAME_MAP, NUM_CLASSES  # Передаем NUM_CLASSES
    )

    print("\n" + "=" * 30 + " Сводка Предсказаний по Файлам " + "=" * 30)
    for file_id, summary in sorted(file_summaries.items()):
        correct_status = "Правильно" if summary["is_correct"] else "Неправильно"
        print(f"\nФайл: {file_id}")
        print(f"  Истинный класс: {summary['true_name']}")
        print(
            f"  Предсказано (M.Vote): {summary['final_prediction_name']} ({correct_status})"
        )
        print(f"  Учтено голосов (кадров): {summary['total_votes_for_file']}")
        print(
            f"  Средняя уверенность учтенных кадров: {summary['average_confidence_for_file']:.4f}"
        )

        # --- Вывод Топ-3 ---
        print("  Топ-3 предсказанных класса (по средней вероятности):")
        if "top_n_predictions" in summary:
            for i, top_pred in enumerate(summary["top_n_predictions"]):
                print(
                    f"    {i+1}. {top_pred['name']}: {top_pred['average_probability']:.4f}"
                )
        else:
            print("    Не удалось определить топ-3.")

    print("=" * 80)

    # --- 8. Генерация и Сохранение Отчета ---
    print("\n--- Отчет о классификации (на отфильтрованных кадрах/окнах) ---")
    if filtered_true_labels and filtered_preds:
        present_labels_frame = np.unique(
            np.concatenate((filtered_true_labels, filtered_preds))
        )
        present_class_names_frame = [
            LABEL_TO_CLASS_NAME_MAP.get(i, f"Label {i}") for i in present_labels_frame
        ]
        try:
            report_dict = classification_report(
                filtered_true_labels,
                filtered_preds,
                labels=present_labels_frame,
                target_names=present_class_names_frame,
                zero_division=0,
                output_dict=True,
            )
            print(
                classification_report(
                    filtered_true_labels,
                    filtered_preds,
                    labels=present_labels_frame,
                    target_names=present_class_names_frame,
                    zero_division=0,
                )
            )
            try:
                report_df = pd.DataFrame(report_dict).transpose()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_df.to_csv(report_path, index=True)
                print(f"\nОтчет сохранен в {report_path}")
            except Exception as e:
                warn(f"Не удалось сохранить отчет в CSV: {e}")
        except Exception as e:
            print(f"Не удалось сгенерировать отчет: {e}")
    else:
        print("Нет данных для генерации отчета.")

    print("\n--- Пайплайн предсказания MLP завершен ---")


# --- Точка входа ---
if __name__ == "__main__":
    feature_bake(
        eval=True,
        make_dir=True,
        parametrization=True,
        extractor_n_frames=True,
    )
    # Запуск основной функции
    main(
        scaler_path=SCALER_PATH,
        model_weights_path=MODEL_WEIGHTS_PATH,
        eval_feature_dir=EVAL_FEATURE_DIR,
        report_path=REPORT_CSV_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,  # Используем глобальное значение
        use_deep_model_flag=USE_DEEP_MODEL,  # Используем глобальное значение
    )
