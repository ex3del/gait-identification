"""
Модуль для загрузки данных признаков, создания последовательностей и разделения.

Содержит функции для:
1. Создания словарей отображения имен классов в метки и обратно.
2. Загрузки .npy файлов признаков (каждый файл - признаки для кадров одного видео).
3. Создания перекрывающихся временных последовательностей заданной длины из кадров.
4. Разделения сгенерированных последовательностей на обучающую и тестовую выборки
   (разделение происходит *внутри* набора последовательностей, полученных из *каждого*
   исходного .npy файла).
"""

import pprint
from pathlib import Path
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import torch

from ...paths.paths import NAMES

# --- Константы ---
TRAIN_RATIO = (
    0.75  # Доля последовательностей из КАЖДОГО файла, идущая в обучающую выборку
)

# --- Создание словарей для меток ---
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

# --- Функция создания последовательностей ---


def create_sequences_from_files(
    feature_dir: Path,
    names_structure: List[Dict[str, Any]],
    class_map: Dict[str, int],
    stride: int,
    seq_length: int,
    train_ratio: float,
    input_size_per_frame: int,  # Добавлен для проверки размерности
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor], List[int]]:
    """
    Загружает .npy файлы признаков, создает перекрывающиеся последовательности
    и разделяет их на обучающую и тестовую выборки для каждого файла.

    Args:
        feature_dir (Path): Директория с файлами признаков .npy (формат [кадры, признаки]).
        names_structure (List[Dict]): Структура данных NAMES.
        class_map (Dict[str, int]): Словарь для отображения имен классов в метки.
        seq_length (int): Длина создаваемых последовательностей (количество кадров).
        stirde (int): С каким шагом мы формируем последовательности кадров
        train_ratio (float): Доля последовательностей из каждого файла для обучения.
        input_size_per_frame (int): Ожидаемое количество признаков на кадр (для проверки).

    Returns:
        Tuple[List[torch.Tensor], List[int], List[torch.Tensor], List[int]]:
            Кортеж: (train_sequences, train_labels, test_sequences, test_labels)
            Каждый элемент в списках *_sequences - это тензор формы [seq_length, input_size_per_frame].
    """
    train_sequences: List[torch.Tensor] = []
    train_labels: List[int] = []
    test_sequences: List[torch.Tensor] = []
    test_labels: List[int] = []

    print(f"Создание последовательностей из {feature_dir}...")
    print(f"Длина последовательности: {seq_length}")
    print(
        f"Соотношение Train/Test для последовательностей из файла: {train_ratio*100:.1f}% / {(1-train_ratio)*100:.1f}%"
    )

    processed_files = 0
    skipped_files = 0

    for class_info in names_structure:
        class_name = class_info.get("class")
        if not class_name or class_name not in class_map:
            # Предупреждение уже будет при создании словаря, здесь можно пропустить тихо
            # warn(f"Пропуск класса '{class_name}': отсутствует или нет в class_map.")
            skipped_files += len(class_info.get("samples", []))
            continue

        label = class_map[class_name]
        print(f"\nОбработка класса: '{class_name}' (метка: {label})")

        for sample in class_info.get("samples", []):
            output_name = sample.get("out")
            if not output_name:
                warn(f"  Пропуск сэмпла в классе '{class_name}': нет ключа 'out'.")
                skipped_files += 1
                continue

            feature_file_path = feature_dir / f"{output_name}.npy"
            # Убрано лишнее сообщение о загрузке каждого файла, чтобы не засорять лог

            try:
                if not feature_file_path.is_file():
                    raise FileNotFoundError(f"Файл не найден: {feature_file_path}")

                # Загрузка данных кадра (предполагаем форму: [кадры, признаки_на_кадр])
                data = np.load(feature_file_path).astype(np.float32)

                # Проверки данных
                if data.ndim != 2:
                    raise ValueError(
                        f"Некорректная размерность данных {data.ndim} (ожидалось 2)."
                    )
                if data.shape[0] < seq_length:
                    raise ValueError(
                        f"Недостаточно кадров ({data.shape[0]} < {seq_length})."
                    )
                if data.shape[1] == 0:
                    raise ValueError("Количество признаков на кадр равно 0.")
                if data.shape[1] != input_size_per_frame:
                    raise ValueError(
                        f"Несоответствие количества признаков: ожидалось {input_size_per_frame}, получено {data.shape[1]}. Проверьте INPUT_SIZE_PER_FRAME."
                    )

                # Создание перекрывающихся последовательностей из одного файла
                file_sequences = []
                num_frames = data.shape[0]
                for i in range(0, num_frames - seq_length + 1, stride):
                    sequence = data[i : i + seq_length, :]
                    file_sequences.append(torch.tensor(sequence, dtype=torch.float32))

                if not file_sequences:
                    # Это не должно произойти при data.shape[0] >= seq_length
                    warn(
                        f"  Не удалось создать последовательности из файла {feature_file_path}."
                    )
                    continue

                # Разделение последовательностей из ЭТОГО файла на train/test
                split_idx = int(len(file_sequences) * train_ratio)
                train_seqs_file = file_sequences[:split_idx]
                test_seqs_file = file_sequences[split_idx:]

                if train_seqs_file:
                    train_sequences.extend(train_seqs_file)
                    train_labels.extend([label] * len(train_seqs_file))
                if test_seqs_file:
                    test_sequences.extend(test_seqs_file)
                    test_labels.extend([label] * len(test_seqs_file))

                processed_files += 1

            except FileNotFoundError as e:
                warn(f"  Ошибка: {e}. Пропуск файла.")
                skipped_files += 1
            except ValueError as e:
                warn(
                    f"  Ошибка данных в файле '{feature_file_path}': {e}. Пропуск файла."
                )
                skipped_files += 1
            except Exception as e:
                warn(
                    f"  Неожиданная ошибка при обработке файла '{feature_file_path}': {e}. Пропуск файла."
                )
                skipped_files += 1

    print(f"\nСоздание последовательностей завершено.")
    print(f"Обработано файлов: {processed_files}")
    print(f"Пропущено файлов/сэмплов: {skipped_files}")
    print(f"Всего обучающих последовательностей: {len(train_sequences)}")
    print(f"Всего тестовых последовательностей: {len(test_sequences)}")

    if not train_sequences or not test_sequences:
        raise RuntimeError(
            "Не удалось создать обучающие или тестовые последовательности. Проверьте данные и пути."
        )

    return train_sequences, train_labels, test_sequences, test_labels
