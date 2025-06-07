"""
Модуль для загрузки данных признаков, создания последовательностей.

Содержит функции для:
1. Создания словарей отображения имен классов в метки и обратно.
2. Загрузки .npy файлов признаков (каждый файл - признаки для кадров одного видео).
3. Создания перекрывающихся временных последовательностей заданной длины из кадров.
"""

import pprint
from pathlib import Path
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import torch

from ...paths.paths import NAMES

# --- Создание словарей для меток согласно Task-2-Training-code.txt ---
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
    # Упорядоченный список имен классов для отчетов/графиков
    CLASS_NAMES_ORDERED: List[str] = [
        LABEL_TO_CLASS_NAME_MAP[i] for i in range(len(CLASS_NAME_TO_LABEL_MAP))
    ]
    NUM_CLASSES = len(CLASS_NAMES_ORDERED)

except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА при создании словаря меток из NAMES: {e}")
    print(
        "Структура NAMES должна быть списком словарей, где каждый значимый словарь содержит ключ 'class'."
    )
    print("Пример NAMES:")
    pprint.pprint(NAMES[:2])
    exit(1)  # Завершаем выполнение, если не можем создать карту меток


def create_sequences_from_files(
    data_path,  # Теперь принимаем _DataPath объект
    sequence_length: int,
    stride: int,
    names: List[Dict[str, Any]],
    class_name_to_label_map: Dict[str, int],
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Загружает .npy файлы признаков и создает перекрывающиеся последовательности.

    Исправленная функция для работы с _DataPath объектом согласно Task-2-Training-code.txt.

    Args:
        data_path (_DataPath): Объект _DataPath с путями к данным.
        sequence_length (int): Длина создаваемых последовательностей (количество кадров).
        stride (int): Шаг для создания перекрывающихся последовательностей.
        names (List[Dict]): Структура данных NAMES.
        class_name_to_label_map (Dict[str, int]): Словарь отображения классов в метки.

    Returns:
        Tuple[List[torch.Tensor], List[int]]:
            Кортеж (sequences, labels) где каждая последовательность имеет форму [seq_length, features].
    """
    sequences: List[torch.Tensor] = []
    labels: List[int] = []

    print(f"Создание последовательностей из {data_path}...")
    print(f"Длина последовательности: {sequence_length}")
    print(f"Шаг (stride): {stride}")

    # Используем свойство FEATURES объекта _DataPath
    feature_dir = data_path.FEATURES

    print(f"Директория с признаками: {feature_dir}")

    if not feature_dir.exists():
        raise FileNotFoundError(f"Директория с признаками не найдена: {feature_dir}")

    processed_files = 0
    skipped_files = 0

    for class_info in names:
        class_name = class_info.get("class")
        if not class_name or class_name not in class_name_to_label_map:
            # Пропускаем записи без класса или неизвестные классы
            skipped_files += len(class_info.get("samples", []))
            continue

        label = class_name_to_label_map[class_name]
        print(f"\nОбработка класса: '{class_name}' (метка: {label})")

        for sample in class_info.get("samples", []):
            output_name = sample.get("out")
            if not output_name:
                warn(f"  Пропуск сэмпла в классе '{class_name}': нет ключа 'out'.")
                skipped_files += 1
                continue

            feature_file_path = feature_dir / f"{output_name}.npy"

            try:
                if not feature_file_path.is_file():
                    warn(f"  Файл не найден: {feature_file_path}")
                    skipped_files += 1
                    continue

                # Загрузка данных (форма: [кадры, признаки])
                data = np.load(feature_file_path).astype(np.float32)

                # Проверки данных
                if data.ndim != 2:
                    warn(
                        f"  Некорректная размерность данных {data.ndim} в {feature_file_path}"
                    )
                    skipped_files += 1
                    continue

                if data.shape[0] < sequence_length:
                    warn(
                        f"  Недостаточно кадров ({data.shape[0]} < {sequence_length}) в {feature_file_path}"
                    )
                    skipped_files += 1
                    continue

                if data.shape[1] == 0:
                    warn(f"  Количество признаков равно 0 в {feature_file_path}")
                    skipped_files += 1
                    continue

                # Создание перекрывающихся последовательностей
                num_frames = data.shape[0]
                file_sequences = []

                for i in range(0, num_frames - sequence_length + 1, stride):
                    sequence = data[i : i + sequence_length, :]
                    file_sequences.append(torch.tensor(sequence, dtype=torch.float32))

                if file_sequences:
                    sequences.extend(file_sequences)
                    labels.extend([label] * len(file_sequences))
                    print(
                        f"  Обработан {output_name}.npy: {len(file_sequences)} последовательностей"
                    )
                    processed_files += 1
                else:
                    warn(
                        f"  Не удалось создать последовательности из {feature_file_path}"
                    )
                    skipped_files += 1

            except Exception as e:
                warn(f"  Ошибка при обработке {feature_file_path}: {e}")
                skipped_files += 1

    print(f"\nСоздание последовательностей завершено.")
    print(f"Обработано файлов: {processed_files}")
    print(f"Пропущено файлов: {skipped_files}")
    print(f"Всего последовательностей: {len(sequences)}")

    if len(sequences) == 0:
        raise RuntimeError(
            f"Не удалось создать последовательности. Проверьте данные в {feature_dir}"
        )

    # Вывод информации о размерности
    if sequences:
        example_shape = sequences[0].shape
        print(f"Форма одной последовательности: {example_shape}")
        print(f"Признаков на кадр: {example_shape[1]}")

    return sequences, labels
