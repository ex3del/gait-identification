"""
Модуль для обработки 3D-координат ключевых точек, извлечения признаков и их предобработки.

Этот модуль загружает предварительно вычисленные 3D-координаты ключевых точек
(обычно из `.npy` файлов), выполняет заполнение пропущенных значений,
применяет различные методы сглаживания и фильтрации, вычисляет производные,
извлекает геометрические признаки (длины сегментов тела, косинусы углов между
сегментами, углы производных и т.д.), и форматирует итоговый массив признаков
для использования в различных моделях машинного обучения.
"""

from pathlib import Path
import os
from typing import TYPE_CHECKING
from enum import Enum, auto
import json

import numpy as np
import scipy
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

from .paths import (
    TRAIN,
    EVAL,
    NAMES,
    GOOD_FEATURES,
)
from ..features.extract_lengths import (
    extract_lengths,
    extract_near_cosines,
    extract_connection_derivative_angles,
    KeypointScheme,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Union


def smoothen(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Применяет сглаживание с использованием окна Ханна (Hanning window).

    Сглаживание выполняется вдоль оси 0.

    Args:
        arr (np.ndarray): Входной массив данных.
                          Ожидается как минимум 1D массив.
        n (int): Размер окна Ханна (должен быть больше 1).

    Returns:
        np.ndarray: Сглаженный массив той же формы, что и входной.
    """
    assert (
        isinstance(n, int) and n > 1
    ), "Размер окна n должен быть целым числом больше 1."
    filt = scipy.signal.windows.hann(n)
    # Применяем свертку вдоль оси 0 для каждого столбца
    return np.apply_along_axis(
        lambda m: np.convolve(m, filt, mode="same") / np.sum(filt),  # Нормализуем окно
        axis=0,
        arr=arr,
    )


def medfilter(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Применяет медианный фильтр.

    Фильтрация выполняется вдоль оси 0.
    Медианный фильтр хорошо подходит для удаления импульсного шума ("соль и перец").

    Args:
        arr (np.ndarray): Входной массив данных.
        n (int): Размер окна медианного фильтра (нечетное целое число больше 1).

    Returns:
        np.ndarray: Отфильтрованный массив той же формы.
    """
    assert (
        isinstance(n, int) and n > 1 and n % 2 != 0
    ), "Размер окна n должен быть нечетным целым числом больше 1."
    return np.apply_along_axis(
        lambda m: medfilt(m, kernel_size=n),  # Используем kernel_size для ясности
        axis=0,
        arr=arr,
    )


def mean_avg(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Применяет фильтр скользящего среднего (простое усреднение в окне).

    Фильтрация выполняется вдоль оси 0.

    Args:
        arr (np.ndarray): Входной массив данных.
        n (int): Размер окна усреднения (должен быть больше 1).

    Returns:
        np.ndarray: Усредненный массив той же формы.
    """
    assert (
        isinstance(n, int) and n > 1
    ), "Размер окна n должен быть целым числом больше 1."
    filt = np.ones((n,), dtype=np.float64) / n
    return np.apply_along_axis(
        lambda m: np.convolve(m, filt, mode="same"),
        axis=0,
        arr=arr,
    )


def decay_filt(arr: np.ndarray, K: float) -> np.ndarray:
    """
    Применяет экспоненциальное сглаживание (фильтр первого порядка с затуханием).

    Фильтрация выполняется вдоль оси 0.
    Формула: r[i] = K * a[i] + (1 - K) * r[i-1], где r - результат, a - входной массив.
    K - коэффициент сглаживания (ближе к 0 - сильнее сглаживание, ближе к 1 - слабее).

    Args:
        arr (np.ndarray): Входной массив данных.
        K (float): Коэффициент сглаживания (0 < K < 1).

    Returns:
        np.ndarray: Сглаженный массив той же формы.
    """
    assert (
        isinstance(K, float) and 0 < K < 1
    ), "Коэффициент K должен быть float в диапазоне (0, 1)."

    def filt1d(a: np.ndarray) -> np.ndarray:
        """Применяет фильтр к 1D массиву."""
        assert a.ndim == 1
        r = np.empty_like(a)
        r[0] = a[0]  # Первое значение остается без изменений
        for i in range(1, a.size):
            r[i] = K * a[i] + (1 - K) * r[i - 1]
        return r

    # Применяем 1D фильтр к каждому столбцу (по оси 0)
    return np.apply_along_axis(filt1d, axis=0, arr=arr)


def derive_first(arr: np.ndarray) -> np.ndarray:
    """
    Вычисляет первую производную с использованием центральной разностной схемы.

    Применяет фильтр [0.5, 0, -0.5] вдоль оси 1.

    Args:
        arr (np.ndarray): Входной массив (например, (время, признаки)).

    Returns:
        np.ndarray: Массив с первыми производными, той же формы.
                    Значения на границах могут быть менее точными из-за режима 'same'.
    """
    filt = np.array([0.5, 0, -0.5])  # Ядро центральной разности
    # Применяем свертку вдоль оси 1
    return np.apply_along_axis(
        lambda m: np.convolve(m, filt, mode="same"),
        axis=1,  # !!! Выполняется по оси признаков, а не времени !!!
        arr=arr,
    )


def derive_second(arr: np.ndarray) -> np.ndarray:
    """
    Вычисляет вторую производную с использованием разностной схемы второго порядка.

    Применяет фильтр [1, -2, 1] вдоль оси 1.

    Args:
        arr (np.ndarray): Входной массив (например, (время, признаки)).

    Returns:
        np.ndarray: Массив со вторыми производными, той же формы.
                    Значения на границах могут быть менее точными.
    """
    filt = np.array([1, -2, 1])  # Ядро для второй производной
    # Применяем свертку вдоль оси 1
    return np.apply_along_axis(
        lambda m: np.convolve(m, filt, mode="same"),
        axis=1,  # !!! Выполняется по оси признаков, а не времени !!!
        arr=arr,
    )


# --- Функции работы с окнами ---
# slow, but ok for now
def sliding(arr: np.ndarray, n: int = 5, step: int = 2) -> np.ndarray:
    """
    Создает массив из скользящих окон с заданным шагом, объединяя данные окна в строку.
    (Версия Игоря).

    Args:
        arr (np.ndarray): Входной 2D массив (кадры, признаки).
        n (int, optional): Размер окна (количество строк в окне). Defaults to 5.
        step (int, optional): Шаг, с которым выбираются строки для окна. Defaults to 2.
                           Обратите внимание: шаг применяется *внутри* окна, а не между окнами.
                           Окно сдвигается на 1 строку за раз.

    Returns:
        np.ndarray: Новый массив формы (num_windows, n * num_features),
                    где num_windows - количество возможных окон.
    """
    assert arr.ndim == 2, "Входной массив должен быть 2D."
    num_frames, num_features = arr.shape
    # Индексы строк внутри одного окна
    window_row_indices = np.arange(n) * step
    # Максимальный индекс строки в последнем окне
    max_last_index = window_row_indices[-1]
    # Количество возможных окон
    num_windows = num_frames - max_last_index

    if num_windows <= 0:
        print(
            f"Предупреждение: Невозможно создать окна с n={n}, step={step} для массива высотой {num_frames}. Возвращен пустой массив."
        )
        return np.empty((0, n * num_features))

    # Создаем массив окон
    # np.stack([...], axis=0) собирает строки в массив
    return np.stack(
        [arr[window_row_indices + i, :].flatten() for i in range(num_windows)],
        axis=0,
    )


def sliding_window(arr: np.ndarray, window_size: int = 30, step: int = 3) -> np.ndarray:
    """
    Применяет скользящее окно к 2D массиву (кадры, признаки).

    Создает новый массив, где каждая строка представляет собой "развернутое"
    (flattened) окно из исходного массива. Окно сдвигается с заданным шагом.

    Args:
        arr (np.ndarray): Входной 2D массив (n_frames, n_features).
        window_size (int, optional): Ширина окна (количество кадров в окне). Defaults to 30.
        step (int, optional): Шаг, с которым окно сдвигается по оси кадров. Defaults to 3.

    Returns:
        np.ndarray: Новый массив формы (num_windows, window_size * n_features).
                    num_windows - количество полных окон, которые можно извлечь.
    """
    assert arr.ndim == 2, "Входной массив должен быть 2D."
    n_frames, n_features = arr.shape

    # Проверка возможности создания хотя бы одного окна
    if n_frames < window_size:
        print(
            f"Предупреждение: Высота массива ({n_frames}) меньше размера окна ({window_size}). Возвращен пустой массив."
        )
        return np.empty((0, window_size * n_features))

    # Вычисляем количество окон, которые поместятся
    num_windows = (n_frames - window_size) // step + 1

    # Используем stride_tricks для эффективного создания окон без копирования данных
    # Рассчитываем байтовые шаги (strides)
    bytes_per_item = arr.itemsize
    row_stride, feature_stride = arr.strides

    # Новые шаги для массива окон:
    # Шаг между окнами (по оси 0 нового массива) = step * row_stride
    # Шаг между строками внутри окна (по оси 1 нового массива) = row_stride
    # Шаг между признаками внутри строки (по оси 2 нового массива) = feature_stride
    new_strides = (step * row_stride, row_stride, feature_stride)

    # Создаем "вид" (view) на исходный массив с новой формой и шагами
    # Форма: (количество_окон, размер_окна, количество_признаков)
    windows_view = np.lib.stride_tricks.as_strided(
        arr, shape=(num_windows, window_size, n_features), strides=new_strides
    )

    # Разворачиваем каждое окно в одну строку
    # reshape(num_windows, -1) автоматически вычисляет размер второго измерения
    # Используем .copy() чтобы создать независимый массив, т.к. windows_view - это view.
    result = windows_view.reshape(num_windows, -1).copy()

    return result


# --- Функция заполнения пропусков ---


def replace_missing_values(arr: np.ndarray) -> np.ndarray:
    """
    Заполняет отсутствующие значения (представленные как 0) в 3D координатах
    ключевых точек, используя значение из предыдущего кадра (forward fill).

    Исключает из обработки точки с индексами 0, 1, 2, 4 ( нос, глаза, уши), так как их нулевые значения могут быть легитимными или
    требовать другой обработки.

    Args:
        arr (np.ndarray): Входной массив 3D координат ключевых точек.
                          Ожидаемая форма: (n_frames, n_keypoints, 3).

    Returns:
        np.ndarray: Массив с заполненными значениями (модифицирует входной массив inplace!).
                    Если вы хотите сохранить оригинал, передайте копию: `replace_missing_values(arr.copy())`.
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Ожидается массив формы (n_frames, n_keypoints, 3), получен {arr.shape}"
        )

    n_frames, n_keypoints, n_coords = arr.shape

    # Индексы ключевых точек, которые НЕ нужно заменять (нос, глаза, уши по схеме COCO)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # В коде исключаются 0, 1, 2, 4
    exclude_rows = {0, 1, 2, 4}  # Индексы, которые пропускаем

    # Итерируем по кадрам, начиная со второго
    for frame_idx in range(1, n_frames):
        # Итерируем по ключевым точкам
        for kp_idx in range(n_keypoints):
            # Пропускаем исключенные точки
            if kp_idx in exclude_rows:
                continue

            # Проверяем, является ли точка нулевой (все координаты равны 0)
            # Проверяем только первую координату для скорости, предполагая, что [0,0,0] - маркер пропуска
            # TODO: Уточнить, является ли [0,0,0] единственным маркером пропуска.
            #       Более надежно проверять все координаты: `if np.all(arr[frame_idx, kp_idx] == 0):`
            if arr[frame_idx, kp_idx, 0] == 0:  # Проверяем только X координату == 0
                # Заменяем все координаты значением из предыдущего кадра
                arr[frame_idx, kp_idx, :] = arr[frame_idx - 1, kp_idx, :]

    return arr  # Возвращаем измененный массив


# -------------------------НОВЫЕ ФУНКЦИИ ПО ЗАПОЛНЕНИЮ ПРОПУСКОВ-----------------------------------------


def interpolate_missing_values(arr: np.ndarray) -> np.ndarray:
    """
    Заполняет отсутствующие значения (представленные как [0, 0, 0]) в 3D координатах
    ключевых точек с помощью линейной интерполяции по оси времени.
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Ожидается массив формы (n_frames, n_keypoints, 3), получен {arr.shape}"
        )
    n_frames, n_keypoints, n_coords = arr.shape
    exclude_rows = {0, 1, 2, 4}
    result = arr.copy()
    for kp_idx in range(n_keypoints):
        if kp_idx in exclude_rows:
            continue
        kp_data = result[:, kp_idx, :]
        missing_mask = np.all(kp_data == 0, axis=1)
        if not np.any(missing_mask):
            continue
        valid_indices = np.where(~missing_mask)[0]
        if len(valid_indices) < 2:
            continue
        for coord_idx in range(n_coords):
            valid_data = kp_data[valid_indices, coord_idx]
            interpolator = interp1d(
                valid_indices,
                valid_data,
                kind="spline",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            missing_indices = np.where(missing_mask)[0]
            interpolated_values = interpolator(missing_indices)
            result[missing_indices, kp_idx, coord_idx] = interpolated_values
    return result


def knn_impute_missing_values(arr: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    Заполняет пропущенные значения в 3D координатах с помощью KNN-импутации.
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Ожидается массив формы (n_frames, n_keypoints, 3), получен {arr.shape}"
        )
    n_frames, n_keypoints, n_coords = arr.shape
    exclude_rows = {0, 1, 2, 4}
    result = arr.copy()
    reshaped = result.reshape(n_frames, -1)
    reshaped_nan = reshaped.copy()
    for kp_idx in range(n_keypoints):
        if kp_idx in exclude_rows:
            continue
        start_col = kp_idx * 3
        end_col = start_col + 3
        missing_mask = np.all(result[:, kp_idx, :] == 0, axis=1)
        reshaped_nan[missing_mask, start_col:end_col] = np.nan
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputed = imputer.fit_transform(reshaped_nan)
    return imputed.reshape(n_frames, n_keypoints, 3)


def moving_average_impute(arr: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Заполняет пропущенные значения с помощью скользящего среднего в окне.
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Ожидается массив формы (n_frames, n_keypoints, 3), получен {arr.shape}"
        )
    n_frames, n_keypoints, n_coords = arr.shape
    exclude_rows = {0, 1, 2, 4}
    result = arr.copy()
    for kp_idx in range(n_keypoints):
        if kp_idx in exclude_rows:
            continue
        kp_data = result[:, kp_idx, :]
        missing_mask = np.all(kp_data == 0, axis=1)
        for frame_idx in np.where(missing_mask)[0]:
            start = max(0, frame_idx - window_size // 2)
            end = min(n_frames, frame_idx + window_size // 2 + 1)
            window_data = kp_data[start:end, :]
            valid_mask = ~np.all(window_data == 0, axis=1)
            valid_data = window_data[valid_mask, :]
            if len(valid_data) > 0:
                result[frame_idx, kp_idx, :] = np.mean(valid_data, axis=0)
    return result


def ml_impute_missing_values(arr: np.ndarray) -> np.ndarray:
    """
    Заполняет пропущенные значения в 3D координатах с помощью итеративной модельной импутации.
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Ожидается массив формы (n_frames, n_keypoints, 3), получен {arr.shape}"
        )
    n_frames, n_keypoints, n_coords = arr.shape
    exclude_rows = {0, 1, 2, 4}
    result = arr.copy()
    reshaped = result.reshape(n_frames, -1)
    reshaped_nan = reshaped.copy()
    for kp_idx in range(n_keypoints):
        if kp_idx in exclude_rows:
            continue
        start_col = kp_idx * 3
        end_col = start_col + 3
        missing_mask = np.all(result[:, kp_idx, :] == 0, axis=1)
        reshaped_nan[missing_mask, start_col:end_col] = np.nan
    imputer = IterativeImputer(
        estimator=DecisionTreeRegressor(), max_iter=10, random_state=0
    )
    imputed = imputer.fit_transform(reshaped_nan)
    return imputed.reshape(n_frames, n_keypoints, 3)


def create_masks(
    keypoints: np.ndarray, scheme: KeypointScheme
) -> tuple[np.ndarray, np.ndarray]:
    """
    Создает маски для точек и связей на основе пропущенных данных.

    Args:
        keypoints (np.ndarray): 3D координаты ключевых точек, форма (n_frames, n_keypoints, 3).
        scheme (KeypointScheme): Схема скелета (например, KeypointScheme._17).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - point_mask: Маска для точек, форма (n_frames, n_keypoints).
            - connection_mask: Маска для связей, форма (n_frames, n_connections).
    """
    if keypoints.ndim != 3 or keypoints.shape[2] != 3:
        raise ValueError(
            f"Ожидается массив формы (n_frames, n_keypoints, 3), получен {keypoints.shape}"
        )

    n_frames, n_keypoints, _ = keypoints.shape
    exclude_rows = {0, 1, 2, 4}  # Нос, глаза, уши

    # Маска для точек: 1 - валидная точка, 0 - пропущена
    point_mask = np.ones((n_frames, n_keypoints), dtype=np.float32)
    for kp_idx in range(n_keypoints):
        if kp_idx in exclude_rows:
            continue
        missing_mask = np.all(keypoints[:, kp_idx, :] == 0, axis=1)
        point_mask[missing_mask, kp_idx] = 0

    # Получаем связи из схемы скелета
    connections = scheme.get_connections()  # Список пар [(kp1, kp2), ...]
    n_connections = len(connections)

    # Маска для связей: 1 - обе точки валидны, 0 - хотя бы одна пропущена
    connection_mask = np.ones((n_frames, n_connections), dtype=np.float32)
    for conn_idx, (kp1, kp2) in enumerate(connections):
        # Связь валидна, только если обе точки валидны
        conn_valid = point_mask[:, kp1] * point_mask[:, kp2]
        connection_mask[:, conn_idx] = conn_valid

    return point_mask, connection_mask


# --- Основная функция обработки ---


def main(
    add_3d_points: bool = False,
    make_dir: bool = True,
    eval: bool = False,
    extractor_n_frames: bool = False,  # включает sliding_window?
    parametrization: bool = False,  # Включает sliding или sliding_window
    add_new_features: bool = False,  # Включает доп. группы признаков
    distances: bool = False,  # Включает производные длин
    grad_ang: bool = False,  # Включает углы градиентов
    conn_derivative_angles: bool = False,  # Включает углы между сегментами и производными
    GCN_TTF: bool = False,  # Формат вывода для GCN (только 3D точки)
    lstm_sequence: bool = False,  # Формат вывода для LSTM (несколько каналов признаков)
    feature_selecting: bool = False,  # отбираем ли лучшие признаки
    imputation_method: str = "",
    use_masks=False,
) -> None:
    """
    Главная функция для загрузки 3D данных, извлечения признаков и их сохранения.

    Загружает 3D координаты, обрабатывает их (заполнение пропусков, сглаживание),
    извлекает набор базовых и дополнительных геометрических признаков (длины, углы,
    их производные), объединяет их и форматирует результат в соответствии с
    выбранными параметрами (например, с применением скользящего окна или
    специфического форматирования для GCN/LSTM).

    Args:
        add_3d_points (bool, optional): Добавлять ли исходные (сглаженные) 3D координаты
                                        (развернутые в вектор) к массиву признаков. Defaults to False.
        make_dir (bool, optional): Создавать ли необходимые директории для выходных файлов.
                                   Defaults to True.
        eval (bool, optional): Использовать ли режим теста (влияет на выбор входных/выходных
                               путей и обрабатываемых сэмплов). Defaults to False.
        extractor_n_frames (bool, optional): Если True и `parametrization`=True, используется
                                             `sliding_window`. Defaults to True.
        parametrization (bool, optional): Применять ли скользящее окно (`sliding` или
                                         `sliding_window`). Defaults to True.
        add_new_features (bool, optional): Включать ли дополнительные группы признаков
                                           (управляемые флагами `distances`, `grad_ang`,
                                           `conn_derivative_angles`). Defaults to False.
        distances (bool, optional): Включать ли производные длин сегментов как признаки.
                                    Требует `add_new_features=True`. Defaults to False.
        grad_ang (bool, optional): Включать ли косинусы углов между производными сегментов.
                                   Требует `add_new_features=True`. Defaults to False.
        conn_derivative_angles (bool, optional): Включать ли углы между сегментами и производными.
                                                Требует `add_new_features=True`. Defaults to False.
        GCN_TTF (bool, optional): Формировать ли выходной массив только из развернутых 3D координат
                                  (для GCN). Переопределяет другие наборы признаков. Defaults to False.
        lstm_sequence (bool, optional): Формировать ли выходной массив в формате (channels, time, features_per_channel)
                                        для LSTM. Переопределяет другие форматы вывода и `parametrization`. Defaults to False.
        feature_selecting (bool, optional): Отбирать ли лучшие признаки лежащие в файле good_features.json. Defaults to False

    Raises:
        RuntimeError: Если входной `.npy` файл пуст или не найден.
        FileNotFoundError: Если входной `.npy` файл не найден.
        AttributeError: Если объекты `TRAIN`/`EVAL` из `paths.py` не имеют нужных атрибутов.
        ValueError: При некорректных входных данных для функций обработки.

    Returns:
        None: Функция сохраняет результаты в `.npy` файлы.
    """
    # Определение базового пути и режима
    BASE = EVAL if eval else TRAIN
    mode_name = "ТЕСТОВЫЙ" if eval else "ОБУЧАЮЩИЙ"
    print(f"--- Запуск извлечения признаков для режима: {mode_name} ---")

    # Выбор схемы скелета (17 точек)
    SCH = KeypointScheme._17
    if make_dir:
        BASE.check_tree()

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Итерация по классам
    for class_info in NAMES:
        class_name = class_info.get("class", "UnknownClass")
        print(f"\n--- Обработка класса: {class_name} ---")

        # Итерация по сэмплам внутри класса
        for sample in class_info.get("samples", []):
            output_name = sample.get(
                "out"
            )  # Имя для выходных файлов (.npy с признаками)
            # Имена .bag файлов нам здесь не нужны, т.к. читаем .npy из KP_3D

            # Фильтрация для режима оценки: пропускаем сэмплы без 'eval'
            # Важно: эта логика относится к ВЫБОРУ сэмплов для обработки в режиме eval,
            # а не к выбору входного файла (т.к. входной файл всегда BASE.KP_3D / f"{on}.npy")
            if eval and not sample.get(
                "eval"
            ):  # Если режим оценки и поле 'eval' пустое
                print(
                    f"  Пропуск сэмпла '{output_name}' (режим eval, поле 'eval' пустое)."
                )
                continue

            # Проверяем наличие имени для выходного/входного файла
            if not output_name:
                print(
                    f"  Предупреждение: Пропущен сэмпл из-за отсутствия ключа 'out': {sample} (в классе {class_name})"
                )
                skipped_count += 1
                continue

            # --- Формирование путей и обработка ---
            try:
                # Путь к входному файлу с 3D координатами
                input_npy_path = BASE.KP_3D / f"{output_name}.npy"
                # Путь к выходному файлу с признаками
                output_features_path = BASE.FEATURES / f"{output_name}.npy"
                # Путь к маске
                output_mask_path = BASE.FEATURES / f"{output_name}_mask.npy"

                print(f"  Обработка сэмпла: '{output_name}'")
                print(f"    Input 3D KP: {input_npy_path}")
                print(f"    Output Features: {output_features_path}")

                # --- Проверки перед запуском ---
                # 2. Существует ли входной файл?
                if not input_npy_path.exists():
                    print(
                        f"    Предупреждение: Входной .npy файл с 3D точками не найден! Пропуск."
                    )
                    skipped_count += 1
                    continue
                if not input_npy_path.is_file():
                    print(
                        f"    Предупреждение: Путь к входному .npy существует, но не является файлом! Пропуск."
                    )
                    skipped_count += 1
                    continue

                # --- Загрузка и основная обработка ---
                print(f"    Загрузка и обработка 3D координат...")
                # Загрузка 3D координат
                keypoints_3d = np.load(input_npy_path)

                # Проверка на пустой файл
                if keypoints_3d.size == 0:
                    print(f"    Ошибка: Входной файл '{input_npy_path}' пуст! Пропуск.")
                    # raise RuntimeError(f"ERROR: empty file '{input_npy_path}'") # Можно раскомментировать для прерывания
                    error_count += 1
                    continue

                # Проверка размерности (ожидаем N, 17, 3)
                if (
                    keypoints_3d.ndim != 3
                    or keypoints_3d.shape[1]
                    != 17  # TODO: криво, нужно привязать к размерам схемы, но пока не учитываем голову, тяжело
                    or keypoints_3d.shape[2] != 3
                ):
                    print(
                        f"    Ошибка: Неожиданная форма входного массива в '{input_npy_path}': {keypoints_3d.shape}. Ожидалось (N, {SCH.value}, 3). Пропуск."
                    )
                    error_count += 1
                    continue

                # 1. Заполнение пропущенных значений (inplace)
                # Передаем копию, если не хотим изменять исходный массив (хотя он только что загружен)

                point_mask, connection_mask = None, None
                if use_masks:
                    point_mask, connection_mask = create_masks(keypoints_3d, SCH)
                    print(
                        f"   Созданы маски: point_mask {point_mask.shape}, connection_mask {connection_mask.shape}"
                    )

                if imputation_method == "interpolation":
                    print("\n\n Использую интреполяцию для заполнения  \n\n")
                    keypoints_filled = interpolate_missing_values(keypoints_3d)
                elif imputation_method == "knn":
                    print("\n\n Использую KNN для заполнения  \n\n")
                    keypoints_filled = knn_impute_missing_values(keypoints_3d)
                elif imputation_method == "moving_average":
                    print("\n\n Использую Moving_Average для заполнения  \n\n")
                    keypoints_filled = moving_average_impute(keypoints_3d)
                elif imputation_method == "ml":
                    print("\n\n Использую ML для заполнения  \n\n")
                    keypoints_filled = ml_impute_missing_values(keypoints_3d)
                else:
                    keypoints_filled = replace_missing_values(keypoints_3d)

                # 2. Первичное сглаживание (экспоненциальное)
                # Параметры сглаживания
                # smooth_radius = 7 # Не используется с decay_filt
                K = 0.007  # Коэффициент для decay_filt
                # keypoints_filled = mean_avg(keypoints_filled, smooth_radius) # Альтернатива
                keypoints_filled_smooth = decay_filt(keypoints_filled, K)

                # 3. Извлечение базовых признаков (углы, расстояния) из СГЛАЖЕННЫХ данных
                # Косинусы углов между смежными сегментами
                coss = np.arccos(extract_near_cosines(keypoints_filled_smooth, SCH))
                # Длины сегментов
                distss = extract_lengths(keypoints_filled_smooth, SCH)

                # 4. Вычисление производных от СГЛАЖЕННЫХ 3D координат
                # Первая производная (скорость точек)
                vects = derive_first(keypoints_filled_smooth)  # Применяется по оси 1!
                # Длины векторов скорости (модуль скорости сегментов)
                distss2 = extract_lengths(vects, SCH)
                # Вторая производная (ускорение точек)
                vects2 = derive_second(keypoints_filled_smooth)  # Применяется по оси 1!
                # Длины векторов ускорения
                distss3 = extract_lengths(vects2, SCH)

                # 5. Дополнительное сглаживание извлеченных признаков
                smooth_radius_features = 5  # Не используется с decay_filt
                K_features = 0.05
                # Функция для сглаживания признаков
                func2 = lambda arr: decay_filt(arr, K_features)

                if use_masks:
                    print("    Применение масок к признакам...")
                    # Маски для длин и углов (зависят от связей)
                    coss = coss * connection_mask
                    distss = distss * connection_mask
                    distss2 = distss2 * connection_mask
                    distss3 = distss3 * connection_mask
                # Сглаживание базовых признаков
                coss = func2(coss)
                distss = func2(distss)
                distss2 = func2(distss2)
                distss3 = func2(distss3)

                # Производные от сглаженных косинусов
                dcos = derive_first(coss)  # Применяется по оси 1!
                ddcos = derive_second(dcos)  # Применяется по оси 1!

                # 6. Расчет дополнительных признаков (если включены)
                feature_arrays = [coss, distss, dcos, ddcos, distss2, distss3]

                if add_3d_points:
                    # Добавляем развернутые сглаженные 3D координаты
                    # Форма keypoints_filled_smooth: (n_frames, 17, 3) -> (n_frames, 51)
                    points_flat = keypoints_filled_smooth.reshape(
                        keypoints_filled_smooth.shape[0], -1
                    )

                    if use_masks:
                        # Маска для точек применяется к развернутым координатам
                        points_mask = np.repeat(
                            point_mask, 3, axis=1
                        )  # (n_frames, n_keypoints * 3)
                        points_flat = points_flat * points_mask
                    feature_arrays.append(points_flat)

                # Блок дополнительных новых признаков
                if add_new_features:
                    if grad_ang:
                        # Косинусы углов между векторами скорости (производными 1-го порядка)
                        grad_ang_1 = func2(np.arccos(extract_near_cosines(vects, SCH)))
                        # Косинусы углов между векторами ускорения (производными 2-го порядка)
                        grad_ang_2 = func2(np.arccos(extract_near_cosines(vects2, SCH)))
                        if use_masks:
                            grad_ang_1 = grad_ang_1 * connection_mask
                            grad_ang_2 = grad_ang_2 * connection_mask
                        feature_arrays.extend([grad_ang_1, grad_ang_2])

                    if distances:
                        # Производные от длин сегментов
                        lengths2 = func2(derive_first(distss))  # Применяется по оси 1!
                        lengths3 = func2(derive_second(distss))  # Применяется по оси 1!
                        if use_masks:
                            lengths2 = lengths2 * connection_mask
                            lengths3 = lengths3 * connection_mask
                        feature_arrays.extend([lengths2, lengths3])

                    if conn_derivative_angles:
                        # Углы между сегментами и производными (скоростью) точек
                        conn_deriv_angles = extract_connection_derivative_angles(
                            keypoints_filled_smooth, vects, SCH
                        )
                        conn_deriv_angles = func2(conn_deriv_angles)
                        if use_masks:
                            conn_deriv_angles = conn_deriv_angles * connection_mask
                        feature_arrays.append(conn_deriv_angles)

                # 7. Объединение всех выбранных признаков
                try:
                    result_array = np.concatenate(
                        feature_arrays, axis=1
                    )  # Объединяем по оси признаков
                    print(f"    Промежуточная форма признаков: {result_array.shape}")
                except ValueError as e:
                    print(f"    Ошибка конкатенации признаков: {e}")
                    print(
                        f"    Формы массивов для конкатенации: {[arr.shape for arr in feature_arrays]}"
                    )
                    error_count += 1
                    continue  # Пропускаем этот сэмпл

                # 8. Специальное форматирование / Постобработка
                if GCN_TTF:
                    # Для GCN оставляем только развернутые 3D координаты
                    print("    Форматирование для GCN...")
                    result_array = keypoints_filled_smooth.reshape(
                        keypoints_filled_smooth.shape[0], -1
                    )
                    if use_masks:
                        points_mask = np.repeat(point_mask, 3, axis=1)
                        result_array = result_array * points_mask

                elif lstm_sequence:
                    # Форматирование для LSTM: (каналы, время, признаки_на_канал)
                    print("    Форматирование для LSTM...")
                    # Целевое количество признаков на канал (например, 16)
                    TARGET_LSTM_FEATURES = 16

                    def pad_features(arr, target_size=TARGET_LSTM_FEATURES):
                        """Дополняет признаки нулями до target_size по оси 1."""
                        if arr.shape[1] < target_size:
                            pad_width = ((0, 0), (0, target_size - arr.shape[1]))
                            return np.pad(
                                arr, pad_width, mode="constant", constant_values=0
                            )
                        elif arr.shape[1] > target_size:
                            # Обрезаем, если признаков больше
                            return arr[:, :target_size]
                        return arr

                    # Приводим все базовые признаки к единому размеру
                    coss_pad = pad_features(coss)
                    distss_pad = pad_features(distss)
                    dcos_pad = pad_features(dcos)
                    ddcos_pad = pad_features(ddcos)
                    distss2_pad = pad_features(distss2)
                    distss3_pad = pad_features(distss3)

                    # Добавляем ось для каналов и объединяем
                    lstm_feature_arrays = [
                        np.expand_dims(coss_pad, axis=0),  # (1, T, 16)
                        np.expand_dims(distss_pad, axis=0),  # (1, T, 16)
                        np.expand_dims(dcos_pad, axis=0),  # (1, T, 16)
                        np.expand_dims(ddcos_pad, axis=0),  # (1, T, 16)
                        np.expand_dims(distss2_pad, axis=0),  # (1, T, 16)
                        np.expand_dims(distss3_pad, axis=0),  # (1, T, 16)
                    ]
                    # Можно добавить другие признаки сюда, если add_new_features=True, аналогично их обработав

                    result_array = np.concatenate(
                        lstm_feature_arrays, axis=0
                    )  # Форма: (num_channels, T, 16)
                    print(f"    Итоговая форма для LSTM: {result_array.shape}")

                elif parametrization:
                    # Применение скользящего окна, если не GCN и не LSTM
                    print("    Применение скользящего окна...")
                    if (
                        result_array.shape[0] < 30
                    ):  # Проверка на минимальную длину для окна
                        print(
                            f"    Предупреждение: Недостаточно кадров ({result_array.shape[0]}) для скользящего окна. Пропуск параметризации."
                        )
                    elif extractor_n_frames:
                        window_size = 30
                        step = 3
                        result_array = sliding_window(
                            result_array, window_size=window_size, step=step
                        )
                        print(
                            f"    Применено sliding_window(window={window_size}, step={step})"
                        )
                    else:
                        window_size_n = 5
                        step_n = 2
                        result_array = sliding(
                            result_array, n=window_size_n, step=step_n
                        )
                        print(
                            f"    Применено sliding(n={window_size_n}, step={step_n})"
                        )

                # 8.1 отсечение признаков
                if feature_selecting:
                    filtered_result_array = result_array[
                        :, GOOD_FEATURES["good_feature_indices"]
                    ]
                    result_array = filtered_result_array
                # 9. Сохранение результата
                print(
                    f"    Сохранение итогового массива признаков формы {result_array.shape} в {output_features_path}"
                )
                np.save(
                    output_features_path, result_array.astype(np.float32)
                )  # Сохраняем как float32

                if use_masks:
                    print(f"    Сохранение маски связей в {output_mask_path}")
                    np.save(output_mask_path, connection_mask.astype(np.float32))

                processed_count += 1
                print(f"    Обработка сэмпла '{output_name}' завершена.")

            except FileNotFoundError as e:
                print(
                    f"    Ошибка: Не найден входной файл для сэмпла '{output_name}': {e}"
                )
                error_count += 1
            except ValueError as e:
                print(
                    f"    Ошибка значения (возможно, проблема с данными) для сэмпла '{output_name}': {e}"
                )
                error_count += 1
            except Exception as e:
                print(
                    f"    Критическая ошибка при обработке сэмпла '{output_name}': {e}"
                )
                import traceback

                traceback.print_exc()
                error_count += 1
            # --- Конец блока try/except для сэмпла ---
        # --- Конец внутреннего цикла по сэмплам ---
    # --- Конец внешнего цикла по классам ---

    print(f"\n--- Статистика извлечения признаков ({mode_name}) ---")
    print(f"Успешно обработано (создано .npy): {processed_count}")
    print(f"Пропущено (уже существует / нет вх. файлов / др.): {skipped_count}")
    print(f"Ошибок: {error_count}")
    print(f"--- Обработка режима {mode_name} завершена ---")


# --- Точка входа ---
if __name__ == "__main__":
    # Пример вызова для обучающего набора с параметрами по умолчанию
    main(
        eval=False,
        make_dir=True,
        add_new_features=False,
        distances=False,
        grad_ang=False,
        conn_derivative_angles=False,
        parametrization=True,
        extractor_n_frames=True,
        feature_selecting=False,
        imputation_method="",
        use_masks=False,
    )
