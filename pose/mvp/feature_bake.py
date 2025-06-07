"""
Модуль для обработки 3D-координат ключевых точек, извлечения признаков и их предобработки.

Этот модуль загружает предварительно вычисленные 3D-координаты ключевых точек
(обычно из `.npy` файлов), выполняет заполнение пропущенных значений,
применяет различные методы сглаживания и фильтрации, вычисляет производные,
извлекает геометрические признаки (длины сегментов тела, косинусы углов между
сегментами, углы производных и т.д.), и форматирует итоговый массив признаков
для использования в различных моделях машинного обучения.
"""

from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor

import hydra
from omegaconf import DictConfig, OmegaConf

from ..features.extract_lengths import (
    KeypointScheme,
    extract_connection_derivative_angles,
    extract_lengths,
    extract_near_cosines,
)
from .paths.paths import EVAL, GOOD_FEATURES, NAMES, TRAIN

if TYPE_CHECKING:
    pass


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


def get_smoothing_function(method: str, **kwargs):
    """
    Возвращает функцию сглаживания на основе метода из конфига.
    """
    if method == "decay_filt":
        K = kwargs.get("K", 0.007)
        return lambda arr: decay_filt(arr, K)
    elif method == "hann":
        window_size = kwargs.get("hann_window_size", 7)
        return lambda arr: smoothen(arr, window_size)
    elif method == "median":
        window_size = kwargs.get("median_window_size", 5)
        return lambda arr: medfilter(arr, window_size)
    elif method == "mean_avg":
        window_size = kwargs.get("mean_avg_window_size", 5)
        return lambda arr: mean_avg(arr, window_size)
    else:
        raise ValueError(f"Неизвестный метод сглаживания: {method}")


def get_imputation_function(method: str, cfg: DictConfig):
    """
    Возвращает функцию импутации на основе метода из конфига.
    """
    if method == "interpolation":
        return interpolate_missing_values
    elif method == "knn":
        n_neighbors = cfg.imputation.knn.n_neighbors
        return lambda arr: knn_impute_missing_values(arr, n_neighbors)
    elif method == "moving_average":
        window_size = cfg.imputation.moving_average.window_size
        return lambda arr: moving_average_impute(arr, window_size)
    elif method == "ml":
        return ml_impute_missing_values
    else:  # method == "" или любой другой
        return replace_missing_values


@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")

# --- Основная функция обработки ---


def main(cfg: DictConfig) -> None:
    """
    Главная функция для загрузки 3D данных, извлечения признаков и их сохранения.

    Args:
        cfg (DictConfig): Конфигурация Hydra со всеми параметрами обработки.

    Returns:
        None: Функция сохраняет результаты в `.npy` файлы.
    """
    print("=== Конфигурация препроцессинга ===")
    print(OmegaConf.to_yaml(cfg.preprocessing))
    print("=" * 50)

    # Извлекаем параметры из конфига
    add_3d_points = cfg.preprocessing.features.add_3d_points
    make_dir = cfg.preprocessing.system.make_dir
    eval_mode = cfg.preprocessing.system.eval_mode
    extractor_n_frames = cfg.preprocessing.output_format.extractor_n_frames
    parametrization = cfg.preprocessing.output_format.parametrization
    add_new_features = cfg.preprocessing.features.add_new_features
    distances = cfg.preprocessing.features.distances
    grad_ang = cfg.preprocessing.features.grad_ang
    conn_derivative_angles = cfg.preprocessing.features.conn_derivative_angles
    GCN_TTF = cfg.preprocessing.output_format.GCN_TTF
    lstm_sequence = cfg.preprocessing.output_format.lstm_sequence
    feature_selecting = cfg.preprocessing.features.feature_selecting
    imputation_method = cfg.preprocessing.imputation.method
    use_masks = cfg.preprocessing.masks.use_masks

    # Определение базового пути и режима
    BASE = EVAL if eval_mode else TRAIN
    mode_name = "ТЕСТОВЫЙ" if eval_mode else "ОБУЧАЮЩИЙ"
    print(f"--- Запуск извлечения признаков для режима: {mode_name} ---")

    # Выбор схемы скелета
    SCH = KeypointScheme._17
    if make_dir:
        BASE.check_tree()

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Получаем множество исключенных ключевых точек из конфига
    exclude_rows = set(cfg.preprocessing.skeleton.exclude_keypoints)

    # Настройка функций сглаживания из конфига
    primary_smooth_func = get_smoothing_function(
        cfg.preprocessing.smoothing.primary.method,
        K=cfg.preprocessing.smoothing.primary.K,
        hann_window_size=cfg.preprocessing.smoothing.hann_window_size,
        median_window_size=cfg.preprocessing.smoothing.median_window_size,
        mean_avg_window_size=cfg.preprocessing.smoothing.mean_avg_window_size,
    )

    feature_smooth_func = get_smoothing_function(
        cfg.preprocessing.feature_smoothing.method,
        K=cfg.preprocessing.feature_smoothing.K,
        window_size=cfg.preprocessing.feature_smoothing.window_size,
    )

    # Настройка функции импутации
    imputation_func = get_imputation_function(imputation_method, cfg)

    # Итерация по классам
    for class_info in NAMES:
        class_name = class_info.get("class", "UnknownClass")
        print(f"\n--- Обработка класса: {class_name} ---")

        # Итерация по сэмплам внутри класса
        for sample in class_info.get("samples", []):
            output_name = sample.get("out")

            # Фильтрация для режима оценки
            if eval_mode and not sample.get("eval"):
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
                # Пути к файлам
                input_npy_path = BASE.KP_3D / f"{output_name}.npy"
                output_features_path = BASE.FEATURES / f"{output_name}.npy"
                output_mask_path = BASE.FEATURES / f"{output_name}_mask.npy"

                print(f"  Обработка сэмпла: '{output_name}'")
                print(f"    Input 3D KP: {input_npy_path}")
                print(f"    Output Features: {output_features_path}")

                # --- Проверки перед запуском ---
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
                keypoints_3d = np.load(input_npy_path)

                # Проверка на пустой файл
                if keypoints_3d.size == 0:
                    print(f"    Ошибка: Входной файл '{input_npy_path}' пуст! Пропуск.")
                    error_count += 1
                    continue

                # Проверка размерности
                if (
                    keypoints_3d.ndim != 3
                    or keypoints_3d.shape[1] != 17
                    or keypoints_3d.shape[2] != 3
                ):
                    print(
                        f"    Ошибка: Неожиданная форма входного массива в '{input_npy_path}': {keypoints_3d.shape}. Ожидалось (N, {SCH.value}, 3). Пропуск."
                    )
                    error_count += 1
                    continue

                # 1. Создание масок (если включено)
                point_mask, connection_mask = None, None
                if use_masks:
                    point_mask, connection_mask = create_masks(keypoints_3d, SCH)
                    print(
                        f"   Созданы маски: point_mask {point_mask.shape}, connection_mask {connection_mask.shape}"
                    )

                # 2. Заполнение пропущенных значений
                if imputation_method:
                    print(f"\n\n Использую {imputation_method} для заполнения  \n\n")
                keypoints_filled = imputation_func(keypoints_3d)

                # 3. Первичное сглаживание (из конфига)
                print(
                    f"    Применение сглаживания: {cfg.preprocessing.smoothing.primary.method}"
                )
                keypoints_filled_smooth = primary_smooth_func(keypoints_filled)

                # 4. Извлечение базовых признаков из СГЛАЖЕННЫХ данных
                coss = np.arccos(extract_near_cosines(keypoints_filled_smooth, SCH))
                distss = extract_lengths(keypoints_filled_smooth, SCH)

                # 5. Вычисление производных от СГЛАЖЕННЫХ 3D координат
                vects = derive_first(keypoints_filled_smooth)
                distss2 = extract_lengths(vects, SCH)
                vects2 = derive_second(keypoints_filled_smooth)
                distss3 = extract_lengths(vects2, SCH)

                # 6. Сглаживание извлеченных признаков (из конфига)
                print(
                    f"    Применение сглаживания признаков: {cfg.preprocessing.feature_smoothing.method}"
                )

                if use_masks:
                    print("    Применение масок к признакам...")
                    coss = coss * connection_mask
                    distss = distss * connection_mask
                    distss2 = distss2 * connection_mask
                    distss3 = distss3 * connection_mask

                # Применяем сглаживание признаков
                coss = feature_smooth_func(coss)
                distss = feature_smooth_func(distss)
                distss2 = feature_smooth_func(distss2)
                distss3 = feature_smooth_func(distss3)

                # Производные от сглаженных косинусов
                dcos = derive_first(coss)
                ddcos = derive_second(dcos)

                # 7. Формирование базового набора признаков
                feature_arrays = [coss, distss, dcos, ddcos, distss2, distss3]

                if add_3d_points:
                    points_flat = keypoints_filled_smooth.reshape(
                        keypoints_filled_smooth.shape[0], -1
                    )
                    if use_masks:
                        points_mask = np.repeat(point_mask, 3, axis=1)
                        points_flat = points_flat * points_mask
                    feature_arrays.append(points_flat)

                # 8. Блок дополнительных новых признаков (из конфига)
                if add_new_features:
                    if grad_ang:
                        grad_ang_1 = feature_smooth_func(
                            np.arccos(extract_near_cosines(vects, SCH))
                        )
                        grad_ang_2 = feature_smooth_func(
                            np.arccos(extract_near_cosines(vects2, SCH))
                        )
                        if use_masks:
                            grad_ang_1 = grad_ang_1 * connection_mask
                            grad_ang_2 = grad_ang_2 * connection_mask
                        feature_arrays.extend([grad_ang_1, grad_ang_2])

                    if distances:
                        lengths2 = feature_smooth_func(derive_first(distss))
                        lengths3 = feature_smooth_func(derive_second(distss))
                        if use_masks:
                            lengths2 = lengths2 * connection_mask
                            lengths3 = lengths3 * connection_mask
                        feature_arrays.extend([lengths2, lengths3])

                    if conn_derivative_angles:
                        conn_deriv_angles = extract_connection_derivative_angles(
                            keypoints_filled_smooth, vects, SCH
                        )
                        conn_deriv_angles = feature_smooth_func(conn_deriv_angles)
                        if use_masks:
                            conn_deriv_angles = conn_deriv_angles * connection_mask
                        feature_arrays.append(conn_deriv_angles)

                # 9. Объединение всех выбранных признаков
                try:
                    result_array = np.concatenate(feature_arrays, axis=1)
                    print(f"    Промежуточная форма признаков: {result_array.shape}")
                except ValueError as e:
                    print(f"    Ошибка конкатенации признаков: {e}")
                    print(
                        f"    Формы массивов для конкатенации: {[arr.shape for arr in feature_arrays]}"
                    )
                    error_count += 1
                    continue

                # 10. Специальное форматирование (из конфига)
                if GCN_TTF:
                    print("    Форматирование для GCN...")
                    result_array = keypoints_filled_smooth.reshape(
                        keypoints_filled_smooth.shape[0], -1
                    )
                    if use_masks:
                        points_mask = np.repeat(point_mask, 3, axis=1)
                        result_array = result_array * points_mask

                elif lstm_sequence:
                    print("    Форматирование для LSTM...")
                    TARGET_LSTM_FEATURES = (
                        cfg.preprocessing.lstm_formatting.target_features_per_channel
                    )

                    def pad_features(arr, target_size=TARGET_LSTM_FEATURES):
                        if arr.shape[1] < target_size:
                            pad_width = ((0, 0), (0, target_size - arr.shape[1]))
                            return np.pad(
                                arr, pad_width, mode="constant", constant_values=0
                            )
                        elif arr.shape[1] > target_size:
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
                        np.expand_dims(coss_pad, axis=0),
                        np.expand_dims(distss_pad, axis=0),
                        np.expand_dims(dcos_pad, axis=0),
                        np.expand_dims(ddcos_pad, axis=0),
                        np.expand_dims(distss2_pad, axis=0),
                        np.expand_dims(distss3_pad, axis=0),
                    ]
                    result_array = np.concatenate(lstm_feature_arrays, axis=0)
                    print(f"    Итоговая форма для LSTM: {result_array.shape}")

                elif parametrization:
                    print("    Применение скользящего окна...")
                    if (
                        result_array.shape[0]
                        < cfg.preprocessing.sliding_window.window_size
                    ):
                        print(
                            f"    Предупреждение: Недостаточно кадров ({result_array.shape[0]}) для скользящего окна. Пропуск параметризации."
                        )
                    elif extractor_n_frames:
                        window_size = cfg.preprocessing.sliding_window.window_size
                        step = cfg.preprocessing.sliding_window.step
                        result_array = sliding_window(
                            result_array, window_size=window_size, step=step
                        )
                        print(
                            f"    Применено sliding_window(window={window_size}, step={step})"
                        )
                    else:
                        window_size_n = cfg.preprocessing.sliding_window.n
                        step_n = cfg.preprocessing.sliding_window.step_sliding
                        result_array = sliding(
                            result_array, n=window_size_n, step=step_n
                        )
                        print(
                            f"    Применено sliding(n={window_size_n}, step={step_n})"
                        )

                # 11. Отсечение признаков (из конфига)
                if feature_selecting:
                    filtered_result_array = result_array[
                        :, GOOD_FEATURES["good_feature_indices"]
                    ]
                    result_array = filtered_result_array

                # 12. Сохранение результата
                print(
                    f"    Сохранение итогового массива признаков формы {result_array.shape} в {output_features_path}"
                )
                np.save(output_features_path, result_array.astype(np.float32))

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

    print(f"\n--- Статистика извлечения признаков ({mode_name}) ---")
    print(f"Успешно обработано (создано .npy): {processed_count}")
    print(f"Пропущено (уже существует / нет вх. файлов / др.): {skipped_count}")
    print(f"Ошибок: {error_count}")
    print(f"--- Обработка режима {mode_name} завершена ---")


if __name__ == "__main__":
    # Запуск с Hydra (параметры берутся из конфига)
    main()
