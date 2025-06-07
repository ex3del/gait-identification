"""
Модуль для загрузки и разделения предварительно обработанных данных признаков.

Этот скрипт загружает массивы признаков (сохраненные как `.npy` файлы),
которые были извлечены на предыдущих этапах (например, из модуля feature_extraction).
Он использует структуру данных `NAMES` для итерации по сэмплам и классам,
загружает соответствующий файл признаков для каждого сэмпла из директории `TRAIN.FEATURES`,
разделяет данные *внутри* каждого файла на обучающую и тестовую части согласно
заданному соотношению (`TRAIN_RATIO`), и агрегирует их в единые NumPy массивы
(X_train, y_train, X_test, y_test) для использования в моделях машинного обучения.

Логика разделения Train/Test:
- Каждый файл `.npy` рассматривается как последовательность данных (например, окон).
- Первые `TRAIN_RATIO` % данных из *каждого* файла попадают в общую обучающую выборку.
- Оставшиеся данные из *каждого* файла попадают в общую тестовую выборку.
- Это НЕ разделение на уровне файлов (т.е. не одни файлы в train, другие в test).

Предполагается, что:
- Файлы признаков `.npy` существуют в директории `TRAIN.FEATURES`.
- Глобальная переменная `NAMES` (импортируемая из `paths`) имеет структуру:
  [{"class": "ClassName", "samples": [{"out": "OutputFileBaseName", ...}, ...]}, ...]
- Передаваемый словарь `class_name_to_label_map` корректно отображает
  имена классов ("ClassName") в целочисленные метки.
"""

import pprint  # Для форматирования вывода словаря в сообщении об ошибке
from typing import Dict, Tuple
from warnings import warn  # Для вывода предупреждений

import numpy as np

from ...paths.paths import NAMES, TRAIN

# --- Константы ---

TRAIN_RATIO = 0.75  # Доля данных из КАЖДОГО файла, идущая в обучающую выборку

# --- Основная функция загрузки данных ---


def load_and_split_data(
    class_name_to_label_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает файлы признаков `.npy`, разделяет данные внутри каждого файла на
    обучающую и тестовую части согласно TRAIN_RATIO и агрегирует их.

    Итерирует по классам и сэмплам из `NAMES`. Для каждого сэмпла загружает
    соответствующий файл признаков (по ключу 'out') из `TRAIN.FEATURES`.
    Данные из каждого файла делятся: первые TRAIN_RATIO строк идут в train,
    остальные - в test. Метка класса назначается на основе `class_name` из `NAMES`
    и словаря `class_name_to_label_map`.

    Args:
        class_name_to_label_map (Dict[str, int]): Словарь, отображающий строковое
            имя класса (ключ 'class' из NAMES) в его целочисленную метку.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Кортеж из четырех
            NumPy массивов:
            1. `X_train`: Данные признаков для обучения (формат: (N_train_samples, N_features)).
            2. `y_train`: Метки классов для обучающих данных (формат: (N_train_samples,)).
            3. `X_test`: Данные признаков для тестирования (формат: (N_test_samples, N_features)).
            4. `y_test`: Метки классов для тестовых данных (формат: (N_test_samples,)).
            Объединение данных выполняется с помощью `np.vstack`, меток - `np.concatenate`.

    Raises:
        FileNotFoundError: Если `.npy` файл для какого-либо сэмпла не найден.
        KeyError: Если имя класса из `NAMES` отсутствует в `class_name_to_label_map`.
        ValueError: Если загруженный массив пуст, или возникает ошибка при vstack/concatenate
                    (например, из-за разного количества признаков в файлах).
        Exception: Другие возможные ошибки при загрузке или обработке NumPy массивов.
    """
    # Списки для накопления данных и меток
    train_data_list: list[np.ndarray] = []
    test_data_list: list[np.ndarray] = []
    train_labels_list: list[np.ndarray] = []
    test_labels_list: list[np.ndarray] = []

    print(f"Загрузка и разделение данных из {TRAIN.FEATURES}...")
    print(
        f"Соотношение Train/Test для каждого файла: {TRAIN_RATIO*100:.1f}% / {(1-TRAIN_RATIO)*100:.1f}%"
    )

    processed_samples = 0
    skipped_samples = 0

    # Итерация по классам из NAMES
    for class_info in NAMES:
        class_name = class_info.get("class")

        # Проверка наличия класса и метки для него
        if not class_name:
            warn(f"Обнаружена запись класса без имени: {class_info}. Пропуск.")
            skipped_samples += len(class_info.get("samples", []))
            continue
        if class_name not in class_name_to_label_map:
            dict_fmted = pprint.pformat(class_name_to_label_map, indent=2)
            msg = f"Имя класса '{class_name}' из NAMES отсутствует в словаре class_name_to_label_map:\n{dict_fmted}"
            raise KeyError(msg)
            # Альтернатива: пропустить класс
            # warn(msg)
            # print(f"Пропуск всех сэмплов для класса '{class_name}', т.к. для него нет метки.")
            # skipped_samples += len(class_info.get("samples", []))
            # continue

        label = class_name_to_label_map[class_name]
        print(f"\nОбработка класса: '{class_name}' (метка: {label})")

        # Итерация по сэмплам внутри класса
        for sample in class_info.get("samples", []):
            output_name = sample.get("out")  # Имя базового файла для .npy

            if not output_name:
                warn(
                    f"  Пропущен сэмпл в классе '{class_name}' из-за отсутствия ключа 'out': {sample}"
                )
                skipped_samples += 1
                continue

            # Формирование пути к файлу признаков
            feature_file_path = TRAIN.FEATURES / f"{output_name}.npy"
            print(f"  Загрузка сэмпла: '{output_name}' из файла {feature_file_path}")

            try:
                # Проверка существования файла (строже, чем просто load)
                if not feature_file_path.is_file():
                    raise FileNotFoundError(
                        f"Файл признаков не найден: {feature_file_path}"
                    )

                # Загрузка массива признаков
                arr = np.load(feature_file_path)

                # Проверка на пустой массив
                if arr.size == 0:
                    raise ValueError(f"Файл '{feature_file_path}' пуст.")
                # Проверка, что массив хотя бы 1D (для len())
                if arr.ndim == 0:
                    raise ValueError(
                        f"Массив в файле '{feature_file_path}' имеет 0 измерений."
                    )

                # --- Логика разделения и добавления---
                split_idx = int(len(arr) * TRAIN_RATIO)

                # Добавляем данные для обучения
                if split_idx > 0:
                    train_data_list.append(arr[:split_idx])
                    train_labels_list.append(np.full(split_idx, label, dtype=np.int32))
                else:
                    warn(
                        f"  В файле '{feature_file_path}' ({len(arr)} строк) недостаточно данных для обучающей выборки (split_idx={split_idx})."
                    )

                # Добавляем данные для тестирования
                if len(arr) - split_idx > 0:
                    test_data_list.append(arr[split_idx:])
                    test_labels_list.append(
                        np.full(len(arr) - split_idx, label, dtype=np.int32)
                    )
                else:
                    warn(
                        f"  В файле '{feature_file_path}' ({len(arr)} строк) недостаточно данных для тестовой выборки (split_idx={split_idx})."
                    )

                processed_samples += 1

            except FileNotFoundError as e:
                warn(f"  Ошибка: {e}. Пропуск сэмпла.")
                skipped_samples += 1
            except ValueError as e:  # Ловит пустой файл или некорректную размерность
                warn(
                    f"  Ошибка данных в файле '{feature_file_path}': {e}. Пропуск сэмпла."
                )
                skipped_samples += 1
            except Exception as e:
                warn(
                    f"  Непредвиденная ошибка при обработке файла '{feature_file_path}': {e}. Пропуск сэмпла."
                )
                skipped_samples += 1

    print("\nЗагрузка завершена.")
    print(f"Обработано сэмплов (файлов): {processed_samples}")
    print(f"Пропущено сэмплов: {skipped_samples}")

    # Проверка, были ли загружены хоть какие-то данные
    if not train_data_list or not test_data_list:
        # Если данных нет, возвращаем пустые массивы
        warn("Не удалось загрузить данные ни для обучающей, ни для тестовой выборки.")
        # Определяем количество признаков (столбцов) из первого непустого
        # массива, если он есть
        num_features = 0
        first_valid_arr = next(
            (arr for arr in train_data_list + test_data_list if arr.size > 0), None
        )
        if first_valid_arr is not None:
            if first_valid_arr.ndim > 1:
                num_features = first_valid_arr.shape[1]
            elif first_valid_arr.ndim == 1:
                num_features = 1

        shape_data = (0, num_features) if num_features > 0 else (0,)
        shape_labels = (0,)
        dtype_data = np.float32  # Стандартный тип для признаков
        dtype_labels = np.int32  # Стандартный тип для меток

        print()
        return (
            np.empty(shape_data, dtype=dtype_data),
            np.empty(shape_labels, dtype=dtype_labels),
            np.empty(shape_data, dtype=dtype_data),
            np.empty(shape_labels, dtype=dtype_labels),
        )
    # Объединение списков массивов
    try:
        # np.vstack ожидает, что все массивы в списке имеют одинаковое количество столбцов
        # и будут стакаться по строкам. Это подходит для вашего случая MLP, где
        # строки - сэмплы.
        X_train = np.vstack(train_data_list).astype(np.float32)
        y_train = np.concatenate(train_labels_list).astype(np.int32)
        X_test = np.vstack(test_data_list).astype(np.float32)
        y_test = np.concatenate(test_labels_list).astype(np.int32)
    except ValueError as e:
        print("\n!!! Ошибка при объединении массивов с помощью vstack/concatenate !!!")
        print(f"Ошибка NumPy: {e}")
        print(
            "Убедитесь, что все загруженные массивы признаков имеют одинаковое количество столбцов (признаков)."
        )
        print("Формы некоторых загруженных обучающих массивов:")
        for i, arr in enumerate(train_data_list[:5]):
            print(f"  {i}: {arr.shape}")
        print("Формы некоторых загруженных тестовых массивов:")
        for i, arr in enumerate(test_data_list[:5]):
            print(f"  {i}: {arr.shape}")
        raise  # Перевыбрасываем ошибку

    print("\nИтоговые размеры массивов:")
    print(f"  X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"  X_test:  {X_test.shape}, dtype: {X_test.dtype}")
    print(f"  y_test:  {y_test.shape}, dtype: {y_test.dtype}")

    return X_train, y_train, X_test, y_test
