# -*- coding: utf-8 -*-
"""
Скрипт для выполнения предсказаний (инференса) с использованием
обученной модели LSTM для классификации походки.

Загружает последовательности признаков, применяет StandardScaler, загружает модель.
Выполняет предсказание для каждой последовательности, вычисляет margin (top1-top2).
Фильтрует предсказания по порогу margin.
Агрегирует отфильтрованные предсказания на уровне файлов, используя
голосование по сумме margins (Margin Sum Voting) для финального решения.
Выводит подробный отчет для каждого файла с топ-3 классами (по средней вероятности)
и итоговый classification_report на уровне файлов.
"""

import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from joblib import load as joblib_load
from torch.utils.data import DataLoader, Dataset

from ...feature_bake import main as feature_bake

# --- Импорт пользовательских модулей ---
try:
    from ...paths.paths import EVAL, MODELS, NAMES
    from .load import CLASS_NAME_TO_LABEL_MAP, LABEL_TO_CLASS_NAME_MAP, NUM_CLASSES
    from .LSTM import (
        FFN_HIDDEN_SIZE,
        HIDDEN_SIZE,
        INPUT_SIZE_PER_FRAME,
        NUM_LAYERS,
        SEQUENCE_LENGTH,
        STRIDE,
        USE_BIDIRECTIONAL,
        USE_FFN_HEAD,
        GaitClassifierLSTM,
        seed_worker,
        set_seed,
    )
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать зависимости: {e}")
    exit(1)

# --- Импорт опциональных библиотек ---
try:
    import pandas as pd
except ImportError:
    pd = None
    warn("Библиотека pandas не установлена. Отчет в CSV сохранен не будет.")

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None
    warn(
        "Scikit-learn не установлена. Финальный classification_report не будет сгенерирован."
    )

if TYPE_CHECKING:
    from pathlib import Path

# === Конфигурация Инференса ===

# --- Пути ---
LSTM_WEIGHTS_DIR = MODELS / "LSTM" / "LSTM_weights"
SCALER_PATH = LSTM_WEIGHTS_DIR / "lstm_scaler.joblib"
MODEL_WEIGHTS_PATH = LSTM_WEIGHTS_DIR / "best_lstm_model.pth"
INFERENCE_FEATURE_DIR = EVAL.FEATURES
REPORT_CSV_PATH = (
    LSTM_WEIGHTS_DIR / "lstm_inference_margin_voting_report.csv"
)  # Новое имя отчета

# --- Параметры инференса (параметры модели импортируются) ---
BATCH_SIZE: int = 256
TOP_K_CLASSES: int = 3
# !!! Порог MARGIN для фильтрации предсказаний ПОСЛЕДОВАТЕЛЬНОСТЕЙ перед агрегацией !!!
MARGIN_THRESHOLD: float = 0.1  # Установите желаемый порог
RANDOM_SEED: int = 42

# --- Определение устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Вывод конфигурации ---
print(f"--- Параметры Инференса ---")
print(f"Используемое устройство: {device}")
print(f"Путь к Scaler: {SCALER_PATH}")
print(f"Путь к весам модели: {MODEL_WEIGHTS_PATH}")
print(f"Директория с данными: {INFERENCE_FEATURE_DIR}")
print(f"Количество классов: {NUM_CLASSES}")
print(f"Длина последовательности: {SEQUENCE_LENGTH}")
print(f"Шаг создания последовательностей: {STRIDE}")
print(f"Признаков на кадр: {INPUT_SIZE_PER_FRAME}")
print(f"Размер скрытого слоя LSTM: {HIDDEN_SIZE}")
print(f"Количество слоев LSTM: {NUM_LAYERS}")
print(f"Bidirectional LSTM: {USE_BIDIRECTIONAL}")
print(f"FFN Head: {USE_FFN_HEAD}")
if USE_FFN_HEAD:
    print(f"  FFN Hidden Size: {FFN_HIDDEN_SIZE}")
print(f"Размер батча: {BATCH_SIZE}")
print(
    f"Порог Margin для фильтрации последовательностей: {MARGIN_THRESHOLD}"
)  # Изменено описание
print(f"Seed: {RANDOM_SEED}")
print("-" * 25)


# === Кастомный датасет для Инференса (без изменений) ===
class InferenceGaitSequenceDataset(Dataset):
    # ... (код класса InferenceGaitSequenceDataset) ...
    def __init__(
        self, sequences: List[torch.Tensor], labels: List[int], file_ids: List[str]
    ):
        if not (len(sequences) == len(labels) == len(file_ids)):
            raise ValueError(...)
        if not sequences:
            warn("Создается пустой датасет для инференса.")
        self.sequences = sequences
        self.labels = labels
        self.file_ids = file_ids
        seq_shape = sequences[0].shape if sequences else "(пусто)"
        print(
            f"Создан Inference датасет с {len(self.sequences)} посл. Форма: {seq_shape}"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        return self.sequences[idx], self.labels[idx], self.file_ids[idx]


# === Функция Загрузки Данных для Инференса (без изменений) ===
def load_sequences_for_inference(
    feature_dir: Path,
    names_structure: List[Dict[str, Any]],
    class_map: Dict[str, int],
    seq_length: int,
    stride: int,
    input_size_per_frame: int,
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Загружает .npy файлы, создает перекрывающиеся последовательности для инференса."""
    # ... (код функции load_sequences_for_inference без изменений) ...
    all_sequences, all_true_labels, all_file_identifiers = [], [], []
    print(
        f"\nЗагрузка и создание последовательностей для инференса из {feature_dir}..."
    )
    processed_files, skipped_files = 0, 0
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
    print(
        f"Создана карта 'имя_файла -> метка' для {len(filename_to_label_map)} файлов из NAMES."
    )
    if not feature_dir.is_dir():
        warn(...)
        return [], [], []
    for filename in sorted(os.listdir(feature_dir)):
        if filename.endswith(".npy"):
            base_name = filename[:-4]
            filepath = feature_dir / filename
            true_label = filename_to_label_map.get(base_name)
            if true_label is None:
                warn(...)
                skipped_files += 1
                continue
            try:
                data = np.load(filepath).astype(np.float32)
                if (
                    data.ndim != 2
                    or data.shape[0] < seq_length
                    or data.shape[1] != input_size_per_frame
                ):
                    raise ValueError(...)
                num_frames = data.shape[0]
                sequences_from_file = []
                for i in range(0, num_frames - seq_length + 1, stride):
                    sequence = data[i : i + seq_length, :]
                    sequences_from_file.append(
                        torch.tensor(sequence, dtype=torch.float32)
                    )
                if sequences_from_file:
                    num_seqs = len(sequences_from_file)
                    all_sequences.extend(sequences_from_file)
                    all_true_labels.extend([true_label] * num_seqs)
                    all_file_identifiers.extend([base_name] * num_seqs)
                    processed_files += 1
                else:
                    warn(...)
                    skipped_files += 1
            except Exception as e:
                warn(f"Ошибка обработки файла {filename}: {e}")
                skipped_files += 1
    print(f"\nСоздание последовательностей завершено.")
    print(...)
    print(...)
    if not all_sequences:
        warn(...)
    return all_sequences, all_true_labels, all_file_identifiers


# === Модифицированная Функция Инференса: Возвращает ВСЕ результаты с MARGIN ===
def run_lstm_inference_with_margin(
    model: nn.Module,
    dataloader: DataLoader,
) -> List[Dict[str, Any]]:
    """
    Выполняет инференс и возвращает список словарей для КАЖДОЙ последовательности,
    включая предсказание, истинную метку, ID файла, margin и вероятности.

    Args:
        model (nn.Module): Обученная LSTM модель.
        dataloader (DataLoader): Загрузчик данных (sequences, true_labels, file_ids).

    Returns:
        List[Dict[str, Any]]: Список словарей, по одному на каждую последовательность.
                               Ключи: 'predicted_label', 'true_label', 'file_id',
                                      'margin', 'top1_prob', 'probabilities'.
    """
    global device
    print("\n--- Запуск инференса LSTM (с расчетом margin) ---")
    model.eval()
    results: List[Dict[str, Any]] = []
    processed_sequences_count = 0

    with torch.no_grad():
        for batch_idx, (sequences, true_labels_batch, file_ids_batch) in enumerate(
            dataloader
        ):
            batch_actual_size = sequences.size(0)
            sequences = sequences.to(device)

            outputs = model(sequences)
            probs_batch = torch.softmax(outputs, dim=1)

            # Расчет Margin
            top_k_probs, top_k_indices = torch.topk(probs_batch, 2, dim=1)
            top_probs = top_k_probs[:, 0]
            # Используем clamp(min=0) для top_k_probs[:, 1], если классов < 2
            second_probs = (
                top_k_probs[:, 1]
                if top_k_probs.size(1) > 1
                else torch.zeros_like(top_probs)
            )
            margins = top_probs - second_probs
            preds = top_k_indices[:, 0]

            # Переносим на CPU и собираем результаты
            preds_cpu = preds.cpu().numpy()
            margins_cpu = margins.cpu().numpy()
            top_probs_cpu = top_probs.cpu().numpy()
            probs_cpu = probs_batch.cpu().numpy()
            true_labels_np = np.array(true_labels_batch)

            for i in range(batch_actual_size):
                results.append(
                    {
                        "predicted_label": preds_cpu[i],
                        "true_label": true_labels_np[i],
                        "file_id": file_ids_batch[i],
                        "margin": margins_cpu[i],
                        "top1_prob": top_probs_cpu[i],
                        "probabilities": probs_cpu[i],  # Полное распределение
                    }
                )

            processed_sequences_count += batch_actual_size
            # if (batch_idx + 1) % 20 == 0: print(...) # Логирование прогресса

    total_sequences = len(dataloader.dataset) if dataloader.dataset else 0
    print(f"Инференс завершен. Обработано последовательностей: {total_sequences}")
    print("-" * 20)

    if total_sequences != len(results):
        warn(...)
    return results


# === НОВАЯ Функция Агрегации и Отчета (стиль MLP, с Margin Sum Voting) ===
def aggregate_and_report_margin_voting(
    all_sequence_results: List[Dict[str, Any]],
    label_to_name_map: Dict[int, str],
    num_classes: int,
    margin_threshold: float = 0.0,  # Порог для фильтрации перед агрегацией
    top_k: int = 3,
) -> Tuple[List[int], List[int], Dict[str, Dict]]:
    """
    Фильтрует результаты по margin, агрегирует по файлам (Margin Sum Voting),
    выводит подробный отчет (стиль MLP) и готовит данные для classification_report.

    Args:
        all_sequence_results: Список словарей с результатами для ВСЕХ последовательностей.
        label_to_name_map: Словарь метка -> имя класса.
        num_classes: Общее количество классов.
        margin_threshold (float): Порог margin для фильтрации последовательностей ПЕРЕД агрегацией.
        top_k (int): Количество топ-классов для вывода в отчете (по средней вероятности).

    Returns:
        Tuple[List[int], List[int], Dict[str, Dict]]: Кортеж:
            - final_true_labels: Список истинных меток на уровне файлов.
            - final_predicted_labels: Список финальных предсказанных меток на уровне файлов.
            - file_summaries: Словарь с подробной сводкой для каждого файла.
    """
    if not all_sequence_results:
        warn("Нет данных для агрегации и отчета.")
        return [], [], {}

    print(
        f"\n--- Агрегация результатов по файлам (Margin Sum Voting, порог margin={margin_threshold}) ---"
    )

    # 1. Фильтрация последовательностей по margin
    filtered_results = [
        res for res in all_sequence_results if res["margin"] >= margin_threshold
    ]
    total_sequences = len(all_sequence_results)
    filtered_count = len(filtered_results)
    if total_sequences > 0:
        print(
            f"Отфильтровано последовательностей (margin >= {margin_threshold:.3f}): {filtered_count} из {total_sequences} ({filtered_count/total_sequences*100:.1f}%)"
        )
    else:
        print("Нет последовательностей для фильтрации.")
    if filtered_count == 0:
        warn("После фильтрации не осталось последовательностей для агрегации.")
        return [], [], {}

    # 2. Группировка отфильтрованных данных по file_id
    # Сохраняем предсказанную метку, margin и все вероятности для каждой отфильтрованной посл.
    file_data = defaultdict(
        lambda: {"preds_margins": [], "probs": [], "true_label": -1}
    )
    for res in filtered_results:
        file_id = res["file_id"]
        if file_data[file_id]["true_label"] == -1:
            file_data[file_id]["true_label"] = res["true_label"]
        # Сохраняем пару (предсказанная_метка, margin) и полное распределение вероятностей
        file_data[file_id]["preds_margins"].append(
            (res["predicted_label"], res["margin"])
        )
        file_data[file_id]["probs"].append(res["probabilities"])

    # 3. Агрегация и вывод отчета
    file_summaries: Dict[str, Dict] = {}
    final_true_labels: List[int] = []
    final_predicted_labels: List[int] = []
    correct_files_count = 0

    print("\n" + "=" * 30 + " Подробный отчет по файлам " + "=" * 30)
    for file_id, data in sorted(file_data.items()):
        true_label = data["true_label"]
        true_name = label_to_name_map.get(true_label, f"Label_{true_label}")
        filtered_sequences_count = len(data["preds_margins"])

        if filtered_sequences_count == 0:
            continue  # Не должно быть, но на всякий случай

        # Считаем сумму margins для каждого класса
        margin_sums = np.zeros(num_classes, dtype=np.float64)
        for pred_label, margin_val in data["preds_margins"]:
            if 0 <= pred_label < num_classes:  # Проверка индекса
                margin_sums[pred_label] += margin_val

        # Определяем финальное предсказание (по max margin sum)
        final_pred_label = np.argmax(margin_sums)
        final_pred_name = label_to_name_map.get(
            final_pred_label, f"Label_{final_pred_label}"
        )
        final_pred_score = margin_sums[final_pred_label]  # Максимальная сумма margin

        # Считаем среднее распределение вероятностей (для топ-K)
        avg_prob_dist = np.mean(data["probs"], axis=0)
        # Находим топ-K по средней вероятности
        top_indices = np.argsort(avg_prob_dist)[-top_k:][::-1]
        top_k_report_lines = []
        top_k_for_summary = []
        for i, k_idx in enumerate(top_indices):
            k_name = label_to_name_map.get(k_idx, f"Label_{k_idx}")
            k_prob = avg_prob_dist[k_idx]
            line = f"    {i+1}. {k_name} : {k_prob:.4f}"  # Формат как в MLP
            top_k_report_lines.append(line)
            top_k_for_summary.append(
                {"label": k_idx, "name": k_name, "avg_prob": k_prob}
            )

        # Проверка корректности
        is_correct = final_pred_label == true_label
        if is_correct:
            correct_files_count += 1
        status_str = "[Правильно]" if is_correct else "[Неправильно]"

        # ----- ВЫВОД ОТЧЕТА ДЛЯ ФАЙЛА -----
        print(f"\nФайл: {file_id}")
        print(f"  Истинный класс: {true_name}")
        print(f"  Предсказано (Margin Sum): {final_pred_name} {status_str}")
        print(
            f"  Итоговый счет (Sum of Margins): {final_pred_score:.4f}"
        )  # Выводим счет
        print(f"  Топ-{top_k} классов (по средней вероятности):")
        for line in top_k_report_lines:
            print(line)
        print(
            f"  Учтено последовательностей (с margin >= {margin_threshold:.3f}): {filtered_sequences_count}"
        )
        # ----- Конец вывода -----

        # Сохраняем данные
        final_true_labels.append(true_label)
        final_predicted_labels.append(final_pred_label)
        file_summaries[file_id] = {
            "true_label": true_label,
            "true_name": true_name,
            "predicted_label": final_pred_label,
            "predicted_name": final_pred_name,
            "predicted_score(margin_sum)": float(final_pred_score),  # Сохраняем счет
            "is_correct": is_correct,
            "top_k_details": top_k_for_summary,
            "margin_sums_per_class": {  # Сохраняем все суммы margin
                label_to_name_map.get(idx, idx): score
                for idx, score in enumerate(margin_sums)
                if score > 0  # Только ненулевые
            },
            "num_sequences_considered": filtered_sequences_count,
        }

    print("=" * (62 + len(f" Подробный отчет по файлам ")))

    # Расчет общей точности
    total_files = len(file_summaries)
    if total_files > 0:
        file_accuracy = correct_files_count / total_files
        print(
            f"\nИтоговая точность на уровне файлов (по Margin Sum Voting): {file_accuracy:.4f} ({correct_files_count}/{total_files})"
        )
    else:
        print("\nНе удалось агрегировать результаты ни для одного файла.")

    return final_true_labels, final_predicted_labels, file_summaries


# === Основная Функция Инференса ===
def main():
    """Главная функция для выполнения инференса LSTM модели."""
    print("--- Запуск скрипта инференса LSTM (Margin Sum Voting) ---")
    set_seed(RANDOM_SEED)
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    print("-" * 25)

    # 1. Загрузка Scaler
    # ... (код загрузки scaler) ...
    print(f"\n--- Загрузка Scaler из {SCALER_PATH} ---")
    try:
        scaler = joblib_load(SCALER_PATH)
        print("Scaler успешно загружен.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке Scaler: {e}")
        traceback.print_exc()
        return

    # 2. Загрузка Модели
    # ... (код загрузки модели, используя импортированные параметры) ...
    print(f"\n--- Загрузка Модели LSTM из {MODEL_WEIGHTS_PATH} ---")
    try:
        model = GaitClassifierLSTM(
            input_size=INPUT_SIZE_PER_FRAME,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            use_bidirectional=USE_BIDIRECTIONAL,
            lstm_dropout=0.0,
            use_ffn_head=USE_FFN_HEAD,
            ffn_hidden_size=FFN_HIDDEN_SIZE,
            ffn_dropout=0.0,
        ).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print("Модель и веса успешно загружены.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке модели: {e}")
        traceback.print_exc()
        return

    # 3. Загрузка Данных для Инференса
    # ... (код загрузки данных) ...
    try:
        sequences, true_labels, file_identifiers = load_sequences_for_inference(
            feature_dir=INFERENCE_FEATURE_DIR,
            names_structure=NAMES,
            class_map=CLASS_NAME_TO_LABEL_MAP,
            seq_length=SEQUENCE_LENGTH,
            stride=STRIDE,
            input_size_per_frame=INPUT_SIZE_PER_FRAME,
        )
        if not sequences:
            print("Нет данных для инференса. Завершение.")
            return
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при загрузке данных: {e}")
        traceback.print_exc()
        return

    # 4. Предобработка (Масштабирование)
    # ... (код масштабирования с обработкой AttributeError) ...
    print("\n--- Масштабирование данных ---")
    try:
        scaled_sequences = [
            torch.tensor(scaler.transform(seq.numpy()), dtype=torch.float32)
            for seq in sequences
        ]
        print(f"Масштабирование {len(scaled_sequences)} последовательностей завершено.")
        del sequences
    except AttributeError as e:
        print(...)
        traceback.print_exc()
        return
    except Exception as e:
        print(...)
        traceback.print_exc()
        return

    # 5. Создание DataLoader
    # ... (код создания DataLoader) ...
    print("\n--- Создание DataLoader для инференса ---")
    try:
        inference_dataset = InferenceGaitSequenceDataset(
            scaled_sequences, true_labels, file_identifiers
        )
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=g,
        )
        print(f"DataLoader создан. Количество батчей: {len(inference_loader)}")
        del scaled_sequences
    except Exception as e:
        print(...)
        traceback.print_exc()
        return

    # 6. Запуск Инференса (сбор всех результатов с margin)
    try:
        all_sequence_results = run_lstm_inference_with_margin(
            model=model,
            dataloader=inference_loader,
        )
    except Exception as e:
        print(...)
        traceback.print_exc()
        return

    # 7. Агрегация, Детальный Отчет и Финальный Report
    if all_sequence_results:
        try:
            # Вызываем НОВУЮ функцию агрегации и отчета
            (
                final_true_labels,
                final_predicted_labels,
                file_summaries,
            ) = aggregate_and_report_margin_voting(
                all_sequence_results=all_sequence_results,
                label_to_name_map=LABEL_TO_CLASS_NAME_MAP,
                num_classes=NUM_CLASSES,
                margin_threshold=MARGIN_THRESHOLD,  # Передаем порог для фильтрации
                top_k=TOP_K_CLASSES,
            )

            # 8. Финальный Classification Report (с исправлением меток L0, L1...)
            if classification_report and final_true_labels:
                print(
                    "\n"
                    + "-" * 20
                    + " Финальный Classification Report (на уровне файлов) "
                    + "-" * 20
                )
                try:
                    # !!! ИСПРАВЛЕНИЕ: Используем только присутствующие метки !!!
                    present_labels = sorted(
                        list(set(final_true_labels) | set(final_predicted_labels))
                    )
                    # Исключаем возможные метки неопределенности (-1), если они есть
                    present_labels = [lbl for lbl in present_labels if lbl >= 0]

                    if (
                        present_labels
                    ):  # Генерируем отчет, только если есть что показывать
                        target_names = [
                            LABEL_TO_CLASS_NAME_MAP.get(lbl, f"Label_{lbl}")[:25]
                            for lbl in present_labels
                        ]

                        report_str = classification_report(
                            final_true_labels,
                            final_predicted_labels,
                            labels=present_labels,  # Используем отфильтрованный список меток
                            target_names=target_names,
                            zero_division=0,
                            digits=3,
                        )
                        print(report_str)
                    else:
                        print(
                            "Нет данных (после агрегации/фильтрации) для генерации отчета."
                        )

                except Exception as e:
                    print(f"Не удалось сгенерировать classification_report: {e}")
            elif not classification_report:
                print(
                    "\nНе удалось сгенерировать Classification Report (sklearn недоступен)."
                )
            else:
                print("\nНет агрегированных результатов для Classification Report.")

            # 9. Сохранение CSV отчета (если pandas доступен и есть данные)
            if pd and file_summaries:
                print(f"\n--- Сохранение детального отчета в {REPORT_CSV_PATH} ---")
                try:
                    report_data_list = []
                    for file_id, summary in file_summaries.items():
                        row = {
                            "file_id": file_id,
                            "true_label": summary["true_label"],
                            "true_name": summary["true_name"],
                            "predicted_label": summary["predicted_label"],
                            "predicted_name": summary["predicted_name"],
                            "predicted_score(margin_sum)": summary[
                                "predicted_score(margin_sum)"
                            ],  # Исправлено
                            "is_correct": summary["is_correct"],
                            "num_sequences_considered": summary[
                                "num_sequences_considered"
                            ],  # Исправлено
                        }
                        # Добавляем топ-K предсказания
                        for i, top_info in enumerate(summary["top_k_details"]):
                            row[f"top_{i+1}_pred_name"] = top_info["name"]
                            row[f"top_{i+1}_pred_avg_prob"] = top_info["avg_prob"]
                        report_data_list.append(row)

                    report_df = pd.DataFrame(report_data_list).sort_values(by="file_id")
                    report_df.to_csv(REPORT_CSV_PATH, index=False, float_format="%.4f")
                    print("Детальный отчет успешно сохранен.")
                except Exception as e:
                    warn(f"Не удалось сохранить CSV отчет: {e}")
            elif not pd:
                print("\nPandas не найден, детальный CSV отчет не будет сохранен.")

        except Exception as e:
            print(...)
            traceback.print_exc()
    else:
        print("\nНет данных после инференса для агрегации и отчетов.")

    print("\n--- Скрипт инференса завершен ---")


# --- Точка входа в скрипт ---
if __name__ == "__main__":
    feature_bake(
        eval=True,
        make_dir=True,
        parametrization=False,
        extractor_n_frames=False,
    )
    main()
