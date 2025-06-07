"""
Модуль для извлечения 3D-координат ключевых точек человека из данных RealSense.

Этот модуль использует синхронизированные потоки глубины и цвета из файла
`.bag` Intel RealSense и 2D-координаты ключевых точек (полученные, например,
с помощью YOLOv8-Pose и сохраненные в `.txt` файле), чтобы вычислить
соответствующие 3D-координаты для каждой ключевой точки.

Результаты сохраняются в бинарный файл NumPy (`.npy`).

Основные зависимости: pyrealsense2, numpy.
Предполагается, что файлы `.bag` и `.txt` с 2D-координатами уже существуют
и синхронизированы по кадрам.
"""

from pathlib import Path
import copy
from typing import TYPE_CHECKING

import pyrealsense2 as rs
import numpy as np

# Импорт конфигурации путей и списка файлов
from .paths.paths import (
    TRAIN,
    EVAL,
    NAMES,
)

if TYPE_CHECKING:
    # Уточняем Union для поддержки старых версий Python, если нужно
    # from typing import Union as TypingUnion
    # Python 3.10+ использует |
    pass

# --- Функции ---


def extract_3d_coordinates_from_keypoints(
    bag_file: str | Path,
    keypoints_file: str | Path,
    output_npy: str | Path,
    # fps: int = 30, # Параметр fps не используется в текущей логике
) -> None:
    """
    Извлекает 3D-координаты из файла 2D-ключевых точек, используя данные глубины из bag-файла.

    Читает кадры из bag-файла RealSense (потоки глубины и цвета),
    сопоставляет их с 2D-координатами ключевых точек из текстового файла
    (предполагается покадровая синхронизация), вычисляет 3D-координаты
    (X, Y, Z в метрах относительно камеры) для каждой 2D-точки с использованием
    данных глубины и внутренних параметров камеры (intrinsics), и сохраняет
    результат в файл `.npy`.

    Формат входного файла `keypoints_file` (.txt):
    - Координаты X и Y для каждой точки на отдельной строке, разделенные пробелом.
    - Кадры разделены одной или несколькими пустыми строками.
    - Предполагается наличие 17 ключевых точек на кадр (на данный момент жестко закодировано).

    Формат выходного файла `output_npy` (.npy):
    - NumPy массив типа float32.
    - Форма массива: (количество_кадров, 17, 3), где 17 - количество ключевых точек,
      а 3 - координаты (X, Y, Z).

    Args:
        bag_file (str | Path): Путь к входному `.bag` файлу RealSense.
        keypoints_file (str | Path): Путь к входному `.txt` файлу с 2D-координатами
                                     ключевых точек (пиксельные координаты).
        output_npy (str | Path): Путь для сохранения выходного `.npy` файла с 3D-координатами.
        # fps (int, optional): Целевая частота кадров (в текущей реализации не используется). Defaults to 30.

    Returns:
        None: Функция ничего не возвращает, но создает файл `output_npy`.

    Raises:
        FileNotFoundError: Если `bag_file` или `keypoints_file` не найдены (хотя явной проверки нет).
        ValueError: Если `bag_file` не может быть открыт RealSense SDK.
        RuntimeError: Если возникают ошибки во время обработки RealSense.
        AssertionError: При неверных типах входных аргументов.

    Примечания:
        - Функция предполагает, что количество блоков с ключевыми точками в `keypoints_file`
          соответствует количеству обрабатываемых кадров в `bag_file`.
        - Для точек, выходящих за границы кадра или имеющих невалидную глубину (например, 0),
          в качестве 3D-координат записывается [0, 0, 0].
        - При ошибках парсинга отдельной точки используется запасной механизм: копирование
          координат этой же точки из предыдущего кадра или [0, 0, 0] для первого кадра.
        - Функция ожидает ровно 17 ключевых точек на кадр; кадры с другим количеством пропускаются.
    """
    # Проверка типов входных аргументов
    assert isinstance(
        bag_file, (str, Path)
    ), f"bag_file должен быть str или Path, получен {type(bag_file)}"
    assert isinstance(
        keypoints_file, (str, Path)
    ), f"keypoints_file должен быть str или Path, получен {type(keypoints_file)}"
    assert isinstance(
        output_npy, (str, Path)
    ), f"output_npy должен быть str или Path, получен {type(output_npy)}"
    # assert isinstance(fps, int) and fps > 0, f"fps должен быть положительным int, получен {fps}"

    bag_file_path = Path(bag_file)
    keypoints_file_path = Path(keypoints_file)
    output_npy_path = Path(output_npy)

    if not bag_file_path.exists():
        raise FileNotFoundError(f"Bag файл не найден: {bag_file_path}")
    if not keypoints_file_path.exists():
        raise FileNotFoundError(
            f"Файл 2D ключевых точек не найден: {keypoints_file_path}"
        )

    print(f"Начало извлечения 3D координат:")
    print(f"  Bag файл: {bag_file_path}")
    print(f"  2D точки: {keypoints_file_path}")
    print(f"  Выходной файл: {output_npy_path}")

    # Инициализация пайплайна и конфигурации RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_device_from_file(str(bag_file_path), repeat_playback=False)
    except RuntimeError as e:
        raise ValueError(f"Не удалось открыть bag файл '{bag_file_path}': {e}")

    # Включаем необходимые потоки
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)  # Цвет нужен для выравнивания

    # Запуск пайплайна
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)  # Обрабатываем так быстро, как возможно

    # Объект для выравнивания кадра глубины к кадру цвета
    # Это важно, чтобы пиксельные координаты (x, y) из цветного кадра (где обычно
    # происходит детекция) соответствовали карте глубины.
    align = rs.align(rs.stream.color)

    # Чтение всех 2D ключевых точек из файла
    try:
        with open(keypoints_file_path, "r") as f:
            # Разделяем файл по пустым строкам (разделители кадров)
            # .strip() убирает лишние пробелы/переводы строк в начале/конце
            # filter(None, ...) убирает пустые строки, которые могут появиться из-за нескольких \n подряд
            keypoints_frames_str = f.read().strip().split("\n\n")
            keypoints_frames_str = list(filter(None, keypoints_frames_str))
            num_keypoint_frames = len(keypoints_frames_str)
            print(f"Найдено {num_keypoint_frames} кадров с 2D точками в файле.")
    except Exception as e:
        pipeline.stop()
        raise IOError(
            f"Ошибка чтения файла 2D ключевых точек '{keypoints_file_path}': {e}"
        )

    frames_3d_list = []  # Список для хранения 3D координат всех кадров
    processed_frame_count = 0  # Счетчик обработанных кадров из bag файла

    try:
        print("Обработка кадров из bag файла и сопоставление с 2D точками...")

        # Итерируемся, пока есть кадры с 2D точками для обработки
        while processed_frame_count < num_keypoint_frames:
            # Ожидаем следующую пару синхронизированных кадров (глубина, цвет)
            # Используем try_wait_for_frames для грациозного выхода в конце файла
            success, frames = pipeline.try_wait_for_frames(
                timeout_ms=300
            )  # 1 секунда таймаут

            if not success:
                print(
                    f"Предупреждение: Не удалось получить кадр из bag файла после кадра {processed_frame_count-1} (возможно, конец файла)."
                )
                # Если кадров в bag меньше, чем блоков в txt, останавливаемся
                break

            # Выравнивание кадра глубины к цветному кадру
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            # color_frame = aligned_frames.get_color_frame() # Сам цветной кадр здесь не нужен

            if not depth_frame:  # or not color_frame:
                print(
                    f"Предупреждение: Пропущен кадр {processed_frame_count}, отсутствует кадр глубины (или цвета)."
                )
                # Важно пропустить и соответствующий блок 2D точек, чтобы сохранить синхронизацию
                processed_frame_count += 1
                continue

            # Получаем внутренние параметры (intrinsics) для кадра глубины
            # Они нужны для функции rs2_deproject_pixel_to_point
            try:
                depth_intrinsics = (
                    depth_frame.profile.as_video_stream_profile().get_intrinsics()
                )
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_height, depth_width = depth_image.shape
            except Exception as e:
                print(
                    f"Ошибка получения данных глубины или intrinsics для кадра {processed_frame_count}: {e}. Пропуск кадра."
                )
                processed_frame_count += 1
                continue

            # Получаем строку с 2D точками для ТЕКУЩЕГО кадра
            keypoints_str_list = (
                keypoints_frames_str[processed_frame_count].strip().split("\n")
            )
            # Убираем пустые строки, если они есть внутри блока
            keypoints_str_list = list(filter(None, keypoints_str_list))
            num_keypoints_in_frame = len(keypoints_str_list)

            # Жесткая проверка на ожидаемое количество точек (например, 17 для COCO)
            EXPECTED_KEYPOINTS = 17
            if num_keypoints_in_frame != EXPECTED_KEYPOINTS:
                print(
                    f"Предупреждение: Кадр {processed_frame_count} содержит {num_keypoints_in_frame} 2D точек, ожидалось {EXPECTED_KEYPOINTS}. Пропуск кадра."
                )
                processed_frame_count += 1
                continue  # Пропускаем этот кадр (и блок 2D точек)

            current_frame_3d_points = []  # 3D точки для текущего кадра

            # Итерация по 2D точкам текущего кадра
            for kp_idx, keypoint_str in enumerate(keypoints_str_list):
                try:
                    # Извлекаем координаты x, y
                    x_2d, y_2d = map(int, keypoint_str.split())

                    # Проверяем, находятся ли координаты в пределах кадра глубины
                    if 0 <= x_2d < depth_width and 0 <= y_2d < depth_height:
                        # Получаем значение глубины в этой точке (в миллиметрах)
                        depth_value_mm = depth_frame.get_distance(
                            x_2d, y_2d
                        )  # Используем get_distance() для точности

                        # Если глубина валидна (больше 0)
                        if depth_value_mm > 0:
                            # Конвертируем 2D пиксель + глубину в 3D точку (в метрах)
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrinsics, [x_2d, y_2d], depth_value_mm
                            )
                            current_frame_3d_points.append(
                                point_3d
                            )  # Добавляем [x, y, z]
                        else:
                            # Глубина невалидна (0), используем плейсхолдер
                            current_frame_3d_points.append([0.0, 0.0, 0.0])
                    else:
                        # Координата 2D точки вне кадра, используем плейсхолдер
                        # print(f"  Точка {kp_idx} ({x_2d},{y_2d}) вне кадра ({depth_width}x{depth_height}) кадра {processed_frame_count}")
                        current_frame_3d_points.append([0.0, 0.0, 0.0])

                except ValueError:
                    print(
                        f"Ошибка парсинга координат для точки {kp_idx} в кадре {processed_frame_count}: '{keypoint_str}'. Используем fallback."
                    )
                    # Fallback: Копируем точку из предыдущего кадра, если возможно
                    if frames_3d_list and len(frames_3d_list[-1]) > kp_idx:
                        current_frame_3d_points.append(
                            copy.deepcopy(frames_3d_list[-1][kp_idx])
                        )
                    else:  # Иначе - нули
                        current_frame_3d_points.append([0.0, 0.0, 0.0])
                except Exception as e:
                    print(
                        f"Неожиданная ошибка обработки точки {kp_idx} в кадре {processed_frame_count}: {e}. Используем fallback."
                    )
                    if frames_3d_list and len(frames_3d_list[-1]) > kp_idx:
                        current_frame_3d_points.append(
                            copy.deepcopy(frames_3d_list[-1][kp_idx])
                        )
                    else:
                        current_frame_3d_points.append([0.0, 0.0, 0.0])

            # Добавляем список 3D точек текущего кадра в общий список
            frames_3d_list.append(current_frame_3d_points)
            processed_frame_count += 1  # Переходим к следующему кадру/блоку

            # Логирование прогресса
            if processed_frame_count % 100 == 0:
                print(
                    f"Обработано кадров: {processed_frame_count} / {num_keypoint_frames}"
                )

    except Exception as e:
        # Ловим другие возможные ошибки во время цикла
        print(f"\nКритическая ошибка во время обработки: {e}")
        import traceback

        traceback.print_exc()  # Печатаем traceback для детальной диагностики

    finally:
        # Останавливаем пайплайн в любом случае
        pipeline.stop()
        print("Пайплайн RealSense остановлен.")

        num_frames_processed_successfully = len(frames_3d_list)
        if num_frames_processed_successfully > 0:
            print(f"Успешно обработано {num_frames_processed_successfully} кадров.")
            print(f"Сохранение 3D координат в {output_npy_path}...")

            # Убедимся, что директория для сохранения существует
            output_dir = output_npy_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Конвертируем список списков в NumPy массив нужной формы и типа
            try:
                frames_3d_array = np.array(frames_3d_list, dtype=np.float32)
                # Проверяем форму перед сохранением
                if (
                    frames_3d_array.ndim == 3
                    and frames_3d_array.shape[1] == EXPECTED_KEYPOINTS
                    and frames_3d_array.shape[2] == 3
                ):
                    np.save(output_npy_path, frames_3d_array)
                    print(
                        f"3D координаты успешно сохранены (форма: {frames_3d_array.shape})."
                    )
                else:
                    print(
                        f"Ошибка: Неожиданная форма массива перед сохранением: {frames_3d_array.shape}. Ожидалось (N, {EXPECTED_KEYPOINTS}, 3). Файл не сохранен."
                    )

            except Exception as e:
                print(
                    f"Ошибка при конвертации в NumPy массив или сохранении файла: {e}"
                )

        else:
            print(
                "Не было успешно обработано ни одного кадра. Выходной файл не создан."
            )


# --- Основная функция ---
def main(make_dir: bool = True, eval: bool = False) -> None:
    """
    Главная функция для пакетного извлечения 3D-координат ключевых точек.

    Итерирует по классам и их сэмплам в `NAMES`. Для каждого сэмпла определяет
    пути к соответствующему `.bag` файлу, файлу с 2D-координатами (`.txt`)
    и целевому файлу для 3D-координат (`.npy`).
    Если файл `.npy` еще не существует, вызывает функцию
    `extract_3d_coordinates_from_keypoints` для его создания.

    Args:
        make_dir (bool, optional): Если True, проверяет/создает необходимые директории
                                   (используя `BASE.check_tree()`). Defaults to True.
        eval (bool, optional): Если True, используется режим оценки: берутся пути из `EVAL`,
                               и обрабатываются только те сэмплы, у которых поле 'eval'
                               не пустое (имя .bag файла берется из 'eval').
                               Если False, используется режим обучения (`TRAIN`), имя .bag
                               файла берется из поля 'bag'. Defaults to False.

    Returns:
        None
    """
    BASE = EVAL if eval else TRAIN
    mode_name = "TEST" if eval else "TRAIN"
    print(f"--- Запуск извлечения 3D координат для режима: {mode_name} ---")

    # Проверка/создание директорий
    if make_dir:
        BASE.check_tree()  # Предполагаем, что check_tree создает и KP_3D

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Итерация по классам
    for class_info in NAMES:
        class_name = class_info.get("class", "UnknownClass")
        print(f"\n--- Обработка класса: {class_name} ---")

        # Итерация по сэмплам внутри класса
        for sample in class_info.get("samples", []):
            output_name = sample.get("out")  # Имя для выходных файлов (.txt, .npy)
            bag_name_train = sample.get("bag")  # Имя .bag для режима обучения
            bag_name_eval = sample.get("eval")  # Имя .bag для режима оценки

            # Определяем имя .bag файла в зависимости от режима
            current_bag_basename = None
            if eval:
                # В режиме оценки обрабатываем только сэмплы с непустым 'eval'
                if bag_name_eval and bag_name_eval.strip():
                    current_bag_basename = bag_name_eval
                else:
                    # print(f"  Пропуск сэмпла '{output_name}' (режим eval, поле 'eval' пустое).")
                    continue  # Пропускаем этот сэмпл
            else:
                # В режиме обучения используем 'bag'
                if bag_name_train and bag_name_train.strip():
                    current_bag_basename = bag_name_train
                else:
                    print(
                        f"  Предупреждение: Пропущен сэмпл '{output_name}' (режим train, отсутствует ключ 'bag')."
                    )
                    skipped_count += 1
                    continue

            # Проверяем наличие имени для выходных файлов
            if not output_name:
                print(
                    f"  Предупреждение: Пропущен сэмпл из-за отсутствия ключа 'out': {sample} (в классе {class_name})"
                )
                skipped_count += 1
                continue

            # --- Формирование путей и обработка ---
            try:
                # Путь к .bag файлу
                bag_file = BASE.BAG / f"{current_bag_basename}.bag"
                # Путь к файлу 2D точек (предполагается, что он уже создан)
                kpp_file = BASE.KP_PIXEL / f"{output_name}.txt"
                # Путь к выходному файлу 3D точек
                kp3d_file = BASE.KP_3D / f"{output_name}.npy"

                print(f"  Обработка сэмпла: '{output_name}'")
                print(f"    Input Bag: {bag_file}")
                print(f"    Input 2D KP: {kpp_file}")
                print(f"    Output 3D KP: {kp3d_file}")

                # --- Проверки перед запуском ---
                # 1. Существует ли выходной файл .npy?
                if kp3d_file.exists():
                    if not kp3d_file.is_file():
                        print(
                            f"    Ошибка: Путь '{kp3d_file}' существует, но не является файлом!"
                        )
                        error_count += 1
                    else:
                        print(f"    Файл 3D точек уже существует. Пропуск.")
                        skipped_count += 1
                    continue  # Переходим к следующему сэмплу

                # 2. Существуют ли входные файлы?
                if not bag_file.exists():
                    print(f"    Предупреждение: Входной .bag файл не найден! Пропуск.")
                    skipped_count += 1
                    continue
                if not kpp_file.exists():
                    print(
                        f"    Предупреждение: Входной .txt файл с 2D точками не найден! Пропуск."
                    )
                    skipped_count += 1
                    continue
                if not bag_file.is_file() or not kpp_file.is_file():
                    print(
                        f"    Предупреждение: Один из входных путей существует, но не является файлом! Пропуск."
                    )
                    skipped_count += 1
                    continue

                # --- Запуск извлечения 3D координат ---
                print(f"    Запуск извлечения 3D координат...")
                extract_3d_coordinates_from_keypoints(bag_file, kpp_file, kp3d_file)
                # Проверим, создался ли файл после вызова функции
                if kp3d_file.exists() and kp3d_file.is_file():
                    processed_count += 1
                    print(f"    Извлечение 3D координат завершено для '{output_name}'.")
                else:
                    print(
                        f"    Ошибка: Файл 3D координат не был создан после вызова функции для '{output_name}'."
                    )
                    error_count += 1

            except FileNotFoundError as e:
                print(
                    f"    Ошибка: Не найден входной файл для сэмпла '{output_name}': {e}"
                )
                error_count += 1
            except ValueError as e:  # Ловим ошибки открытия bag файла
                print(
                    f"    Ошибка значения (возможно, проблема с bag файлом) для сэмпла '{output_name}': {e}"
                )
                error_count += 1
            except IOError as e:  # Ловим ошибки чтения txt файла
                print(
                    f"    Ошибка ввода/вывода (возможно, проблема с txt файлом) для сэмпла '{output_name}': {e}"
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

    print(f"\n--- Статистика извлечения 3D координат ({mode_name}) ---")
    print(f"Успешно обработано (создано .npy): {processed_count}")
    print(f"Пропущено (уже существует / нет вх. файлов / др.): {skipped_count}")
    print(f"Ошибок: {error_count}")
    print(f"--- Обработка режима {mode_name} завершена ---")


if __name__ == "__main__":
    # Запуск для обучающего набора
    main(make_dir=True, eval=False)
