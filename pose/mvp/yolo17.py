"""
Модуль для извлечения 2D-координат ключевых точек поз человека из видеофайлов
с использованием модели YOLO и сохранения их в текстовые файлы.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Union

from ultralytics import YOLO

# Импорт конфигурации путей и списка файлов
from .paths.paths import EVAL, NAMES, TRAIN

if TYPE_CHECKING:
    pass


def extract_keypoints_to_txt(
    model_path: Union[str, Path],
    video_path: Union[str, Path],
    output_txt: Union[str, Path],
    conf: float = 0.3,
) -> None:
    """
    Извлекает ключевые точки поз человека из видео с помощью модели YOLO и сохраняет их в текстовый файл.

    Загружает указанную модель YOLO, обрабатывает видеофайл, находит ключевые точки поз
    людей на каждом кадре и записывает их XY-координаты (в пикселях) в выходной файл.

    Формат выходного файла:
    - Каждая строка содержит X и Y координаты одной ключевой точки, разделенные пробелом ("x y").
    - Ключевые точки для всех обнаруженных людей в одном кадре записываются последовательно.
    - Пустая строка вставляется после обработки каждого кадра (даже если на кадре не было найдено поз).

    Args:
        model_path (Union[str, Path]): Путь к файлу модели YOLO (например, 'yolov11x-pose.pt').
        video_path (Union[str, Path]): Путь к входному видеофайлу (например, '.mp4').
        output_txt (Union[str, Path]): Путь к текстовому файлу для сохранения координат ключевых точек.
        conf (float, optional): Порог уверенности для обнаружения поз. Defaults to 0.3.

    Returns:
        None: Функция ничего не возвращает, но создает или перезаписывает `output_txt`.
    """
    # Загрузка модели YOLO для оценки поз
    model = YOLO(model_path)

    # Запуск модели на видеофайле
    # show=False отключает отображение окна с результатами в реальном времени
    results = model(
        source=video_path, conf=conf, show=False, stream=True
    )  # stream=True для экономии памяти

    print(f"Начало обработки видео: {video_path}")
    # Открытие выходного файла для записи
    with open(output_txt, "w") as file:
        frame_count = 0
        keypoint_count_total = 0
        # Итерация по результатам для каждого кадра
        for frame_idx, result in enumerate(results):
            frame_count += 1
            keypoints_found_in_frame = False
            # Проверка наличия ключевых точек в результате обработки кадра
            # result.keypoints может быть None, если ничего не обнаружено
            if result.keypoints is not None:
                # Получение координат (x, y) в виде массива NumPy (переносим на CPU)
                # .xy содержит координаты [x, y] для каждой точки
                # Формат: [количество_людей, количество_ключевых_точек, 2]
                keypoints_array = result.keypoints.xy.cpu().numpy()

                # Если массив не пустой (т.е. обнаружены люди)
                if keypoints_array.size > 0:
                    keypoints_found_in_frame = True
                    # Итерация по каждому обнаруженному человеку в кадре
                    for person_keypoints in keypoints_array:
                        # Итерация по всем ключевым точкам для данного человека
                        for keypoint in person_keypoints:
                            # Преобразование координат в целые числа и запись в файл
                            x, y = int(keypoint[0]), int(keypoint[1])
                            file.write(f"{x} {y}\n")
                            keypoint_count_total += 1

            # Запись пустой строки после обработки данных кадра (даже если точек не было)
            # Это служит разделителем между кадрами в выходном файле
            file.write("\n")

    print(f"Обработка завершена. Всего обработано кадров: {frame_count}.")
    print(f"Всего ключевых точек сохранено: {keypoint_count_total}.")
    print(f"Ключевые точки сохранены в файл: {output_txt}")


def main(make_dir: bool = True, eval: bool = False) -> None:
    """
    Главная функция для пакетной обработки видеофайлов и извлечения ключевых точек.

    Определяет набор данных (обучающий или тестовый) на основе флага `eval`.
    Итерирует по списку записей `NAMES`. Для каждой записи формирует пути
    к видеофайлу (.mp4) и выходному текстовому файлу (.txt).
    Если текстовый файл с ключевыми точками еще не существует, вызывает функцию
    `extract_keypoints_to_txt` для его создания.

    Предполагается, что:
    - Видеофайлы (.mp4) уже существуют в соответствующих директориях (`TRAIN` или `EVAL`).
    - Модуль `paths.py` корректно настроен и содержит объекты `TRAIN`, `EVAL`
      (с атрибутами `MP4`, `KP_PIXEL` и методом `check_tree()`) и список `NAMES`.
    - Модель YOLO (`yolo11x-pose.pt` по умолчанию) доступна по указанному пути.

    Args:
        make_dir (bool, optional): Если True, вызывает `BASE.check_tree()` для создания
                                   необходимых директорий (включая для .txt файлов).
                                   Defaults to True.
        eval (bool, optional): Если True, используется набор данных для оценки (`EVAL`)
                               и обрабатываются только те записи из `NAMES`, у которых
                               поле 'eval' не пустое. Если False, используется
                               обучающий набор (`TRAIN`). Defaults to False.
    Returns:
        None
    """
    # Определение базового пути в зависимости от режима (TRAIN или EVAL)
    BASE = EVAL if eval else TRAIN
    mode_name = "TEST" if eval else "TRAIN"
    print(f"--- Запуск обработки для режима: {mode_name} ---")

    # Создание директорий, если требуется
    if make_dir:
        print("Проверка и создание дерева директорий...")
        BASE.check_tree()

    model_path = "yolo11x-pose.pt"  # Убедитесь, что модель существует по этому пути

    # Итерация по списку записей (словарей) из файла конфигурации
    processed_count = 0
    skipped_count = 0
    error_count = 0
    for class_info in NAMES:
        class_name = class_info.get("class", "UnknownClass")
        print(f"\n--- Обработка класса: {class_name} ---")

        # Внутренний цикл: итерация по сэмплам внутри класса
        for sample in class_info.get("samples", []):  # Используем .get для безопасности
            # Получаем данные ИЗ СЛОВАРЯ sample
            output_name = sample.get("out")
            eval_marker = sample.get("eval")
            bag_name = sample.get("bag", "UnknownBag")  # Для информации

            # Пропускаем запись, если включен режим оценки и поле 'eval' у сэмпла пустое
            if eval and (eval_marker is None or eval_marker == ""):
                # print(f"  Пропуск сэмпла (режим eval, eval пуст): {sample}")
                continue

            # Проверяем, есть ли имя выходного файла у сэмпла
            if not output_name:
                print(
                    f"  Предупреждение: Пропущен сэмпл из-за отсутствия ключа 'out': {sample} (в классе {class_name})"
                )
                skipped_count += 1
                continue

            try:
                # Формирование путей к видео и текстовому файлу
                video_path = BASE.MP4 / f"{output_name}.mp4"
                txt_file = BASE.KP_PIXEL / f"{output_name}.txt"

                print(f"\nОбработка записи: '{output_name}'")
                print(f"  Видеофайл: {video_path}")
                print(f"  Файл точек: {txt_file}")

                # Проверяем существование видеофайла перед обработкой
                if not video_path.exists():
                    print(f"  Предупреждение: Видеофайл не найден! Пропуск.")
                    skipped_count += 1
                    continue
                if not video_path.is_file():
                    print(
                        f"  Предупреждение: Путь к видео существует, но это не файл! Пропуск."
                    )
                    skipped_count += 1
                    continue

                # Проверяем, существует ли уже файл с ключевыми точками
                if not txt_file.exists():
                    print(f"  Файл точек не найден. Запуск извлечения...")
                    # Вызов функции извлечения ключевых точек
                    extract_keypoints_to_txt(model_path, video_path, txt_file)
                    processed_count += 1
                else:
                    # Если файл существует, проверяем, что это действительно файл
                    if not txt_file.is_file():
                        # Если это не файл (например, директория), вызываем ошибку
                        error_count += 1
                        # Можно не прерывать весь процесс, а только вывести ошибку
                        print(
                            f"  Ошибка: Путь '{txt_file}' существует, но не является файлом!"
                        )
                        # raise RuntimeError(f"Path '{txt_file}' exists but is not a file!'") # Раскомментировать для прерывания
                    else:
                        # Файл уже существует, пропускаем обработку
                        print(f"  Файл точек уже существует. Пропуск.")
                        skipped_count += 1

            except Exception as e:
                # Обработка других возможных ошибок во время цикла
                print(f"  Критическая ошибка при обработке записи '{output_name}': {e}")
                error_count += 1

    print(f"\n--- Статистика обработки ({mode_name}) ---")
    print(f"Успешно обработано (создано .txt): {processed_count}")
    print(f"Пропущено (уже существует / нет видео / др.): {skipped_count}")
    print(f"Ошибок: {error_count}")
    print(f"--- Обработка режима {mode_name} завершена ---")


if __name__ == "__main__":
    # Запуск основной функции для обучающего набора данных
    # (make_dir=True - создать директории, eval=False - режим обучения)
    main(make_dir=True, eval=False)
