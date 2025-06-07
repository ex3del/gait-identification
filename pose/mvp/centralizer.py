"""
Скрипт для центрирования 3D координат человеческой позы относительно центра торса,
используя структуру NAMES для определения обрабатываемых файлов.

Загружает .npy файлы с 3D координатами позы (формат: [n_frames, 17, 3]) из
директории TRAIN.KP_3D или EVAL.KP_3D (в зависимости от флага eval_mode),
вычисляет центр торса для каждого кадра, центрирует координаты и сохраняет
результат в соответствующую директорию TRAIN.CENTERED_KP_3D или EVAL.CENTERED_KP_3D.

В режиме оценки (eval_mode=True) обрабатываются только те сэмплы из NAMES,
у которых задано непустое значение в поле 'eval'.
"""

import numpy as np
import os
from pathlib import Path
import traceback  # Для подробного вывода ошибок
from warnings import warn  # Используем warn для некритичных сообщений

# --- Импорт пользовательских модулей ---
try:
    from .paths.paths import TRAIN, EVAL, NAMES
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать зависимости из paths: {e}")
    print("Убедитесь, что структура проекта верна и файл paths.py доступен.")
    exit()


# === КОНФИГУРАЦИЯ ===
EVAL_MODE = False
# --------------------


def center_poses_using_names(eval_mode: bool):
    """
    Обрабатывает .npy файлы с 3D позами, центрируя их относительно торса,
    используя структуру NAMES и флаг eval_mode для выбора данных.

    Args:
        eval_mode (bool): Если True, обрабатываются данные из EVAL.KP_3D
                          (только сэмплы с непустым полем 'eval' в NAMES).
                          Если False, обрабатываются данные из TRAIN.KP_3D.
    """
    # Определяем базовую директорию и режим
    BASE = EVAL if eval_mode else TRAIN
    mode_name = "ОЦЕНОЧНЫЙ" if eval_mode else "ОБУЧАЮЩИЙ"
    input_base_dir = BASE.KP_3D  # Директория с исходными 3D позами
    output_base_dir = BASE.FEATURES  # Директория для сохранения центрированных поз

    print(f"--- Начало центрирования поз ({mode_name} режим) ---")
    print(f"Входная база: {input_base_dir}")
    print(f"Выходная база: {output_base_dir}")

    # Проверка существования входной директории
    if not input_base_dir.is_dir():
        print(f"ОШИБКА: Входная директория не найдена: {input_base_dir}")
        return

    # Создание выходной директории
    try:
        output_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Выходная директория создана или уже существует: {output_base_dir}")
    except Exception as e:
        print(f"ОШИБКА: Не удалось создать выходную директорию {output_base_dir}: {e}")
        return

    # Индексы ключевых точек торса
    TORSO_KEYPOINT_INDICES = [5, 6, 11, 12]  # L_Shoulder, R_Shoulder, L_Hip, R_Hip
    NUM_EXPECTED_KEYPOINTS = 17
    EXPECTED_DIMS = 3

    processed_count = 0
    skipped_count = 0
    error_count = 0
    error_log = []

    # Итерация по классам из NAMES
    for class_info in NAMES:
        class_name = class_info.get("class", "UnknownClass")
        print(f"\n--- Обработка класса: {class_name} ---")

        # Итерация по сэмплам внутри класса
        for sample in class_info.get("samples", []):
            output_name = sample.get(
                "out"
            )  # Имя для входного/выходного .npy файла (без расширения)

            # Фильтрация для режима оценки
            if eval_mode and not sample.get("eval"):
                # print(f"  Пропуск сэмпла '{output_name}' (режим eval, поле 'eval' пустое).") # Детальный лог
                skipped_count += 1
                continue

            # Проверка наличия имени файла
            if not output_name:
                warn(
                    f"  Пропущен сэмпл из-за отсутствия ключа 'out' в классе {class_name}: {sample}"
                )
                skipped_count += 1
                continue

            # Формирование путей
            input_npy_path = input_base_dir / f"{output_name}.npy"
            output_npy_path = output_base_dir / f"{output_name}.npy"

            print(f"  Обработка сэмпла: '{output_name}'")
            # print(f"    Input: {input_npy_path}") # Детальный лог
            # print(f"    Output: {output_npy_path}") # Детальный лог

            # --- Обработка файла ---
            try:
                # 1. Проверка существования входного файла
                if not input_npy_path.is_file():
                    raise FileNotFoundError(f"Входной файл не найден: {input_npy_path}")

                # 2. Загрузка данных
                data = np.load(input_npy_path)
                # print(f"    Загружены данные формы: {data.shape}") # Детальный лог

                # 3. Проверка корректности данных
                if data.ndim != 3:
                    raise ValueError(
                        f"Некорректная размерность ({data.ndim}), ожидалось 3."
                    )
                n_frames, n_keypoints, n_dims = data.shape
                if n_keypoints != NUM_EXPECTED_KEYPOINTS:
                    raise ValueError(
                        f"Некорректное кол-во точек ({n_keypoints}), ожидалось {NUM_EXPECTED_KEYPOINTS}."
                    )
                if n_dims != EXPECTED_DIMS:
                    raise ValueError(
                        f"Некорректное кол-во координат ({n_dims}), ожидалось {EXPECTED_DIMS}."
                    )
                if max(TORSO_KEYPOINT_INDICES) >= n_keypoints:
                    raise ValueError(
                        f"Недостаточно точек ({n_keypoints}) для расчета центра торса (макс. индекс {max(TORSO_KEYPOINT_INDICES)})."
                    )
                if np.isnan(data).any():
                    num_nan = np.isnan(data).sum()
                    warn(f"    Обнаружено {num_nan} NaN. Используется nanmean.")

                # 4. Вычисление центра торса
                torso_points = data[:, TORSO_KEYPOINT_INDICES, :]
                torso_center_per_frame = np.nanmean(torso_points, axis=1)
                if np.isnan(torso_center_per_frame).all():
                    raise ValueError(
                        "Не удалось вычислить центр торса (все точки торса - NaN?)."
                    )

                # 5. Центрирование координат
                centered_data = data - torso_center_per_frame[:, np.newaxis, :]
                # print(f"    Данные центрированы.") # Детальный лог

                # 6. Сохранение результата
                np.save(output_npy_path, centered_data)
                # print(f"    Результат сохранен.") # Детальный лог

                processed_count += 1

            except FileNotFoundError as e:
                error_msg = f"Ошибка для '{output_name}': {e}"
                print(f"    {error_msg}")  # Печатаем ошибку для конкретного файла
                error_log.append(error_msg)
                skipped_count += 1  # Считаем как пропущенный
            except ValueError as e:
                error_msg = f"Ошибка данных для '{output_name}': {e}"
                print(f"    {error_msg}")
                error_log.append(error_msg)
                error_count += 1  # Считаем как ошибку данных
                skipped_count += 1  # И как пропущенный
            except Exception as e:
                error_msg = f"Неожиданная ошибка для '{output_name}': {e}"
                print(f"    {error_msg}")
                traceback.print_exc()  # Полный стектрейс
                error_log.append(error_msg)
                error_count += 1
                skipped_count += 1

    # Вывод итоговой статистики
    print("\n" + "=" * 30 + " Итоги Обработки " + "=" * 30)
    # Считаем общее количество сэмплов, которые должны были быть обработаны
    total_samples_in_names = sum(len(ci.get("samples", [])) for ci in NAMES)
    potential_samples_to_process = 0
    for ci in NAMES:
        for s in ci.get("samples", []):
            if not eval_mode or s.get(
                "eval"
            ):  # Считаем, если не eval режим или если eval режим и есть флаг eval
                if s.get("out"):  # И есть имя файла
                    potential_samples_to_process += 1

    print(f"Режим обработки: {mode_name}")
    print(f"Всего сэмплов в NAMES: {total_samples_in_names}")
    print(
        f"Сэмплов, подлежащих обработке в этом режиме: {potential_samples_to_process}"
    )
    print(f"Успешно обработано и сохранено: {processed_count}")
    print(
        f"Пропущено (файл не найден / нет 'eval' / нет 'out'): {skipped_count - error_count}"
    )  # Вычитаем ошибки, чтобы не считать дважды
    print(f"Ошибки обработки (данные/неожиданные): {error_count}")

    if error_log:
        print("\nСписок ошибок:")
        for i, error in enumerate(error_log):
            print(f"  {i+1}. {error}")
    print("=" * (60 + len(" Итоги Обработки ")))


# --- Точка входа для запуска скрипта ---
if __name__ == "__main__":
    # Вызов основной функции с заданным режимом (EVAL_MODE)
    center_poses_using_names(eval_mode=True)
