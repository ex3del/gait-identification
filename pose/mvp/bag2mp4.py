# ИЗВЛЕЧЕНИЕ RGB ИЗ REALSENSE BAG-ФАЙЛОВ
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pyrealsense2 as rs

from .paths.paths import EVAL, NAMES, TRAIN

if TYPE_CHECKING:
    from os import PathLike
    from typing import Union


def extract_rgb_to_mp4(
    bag_file: "Union[str, PathLike]",
    mp4_file: "Union[str, PathLike]",
    show: bool = False,
):
    """Извлекает RGB-поток из bag-файла Intel RealSense и сохраняет его в MP4-видео.

    Args:
        bag_file (Union[str, PathLike]): Путь к bag-файлу.
        mp4_file (Union[str, PathLike]): Путь для сохранения MP4-видео.
        show (bool, optional): Отображать ли видеопоток в реальном времени. Defaults to False.

    Raises:
        RuntimeError: Если возникает ошибка во время работы с библиотекой RealSense.
        Exception: Если возникает любая другая ошибка.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Включаем устройство из .bag файла
    config.enable_device_from_file(str(bag_file), repeat_playback=False)

    try:
        # Запускаем пайплайн с конфигурацией
        profile = pipeline.start(config)

        # Получаем профиль цветового потока
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)

        color_stream = profile.get_stream(rs.stream.color)
        video_profile = color_stream.as_video_stream_profile()
        fps = video_profile.fps()
        color_intrinsics = video_profile.get_intrinsics()
        width, height = color_intrinsics.width, color_intrinsics.height

        # Выводим обнаруженный FPS
        print(f"Обнаружено FPS: {fps}")

        # Настраиваем OpenCV video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(mp4_file, fourcc, fps, (width, height))

        print("Извлечение RGB потока в MP4...")
        while True:
            # Пытаемся получить кадры без блокировки и исключения при таймауте
            success, frames = pipeline.try_wait_for_frames(
                timeout_ms=150
            )  # Можно использовать меньший таймаут

            # Если кадры не получены (конец файла), выходим из цикла
            if not success:
                print("Кадры больше не поступают, завершение обработки файла.")
                break

            # Если кадры получены, продолжаем обработку
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # ... (остальная часть обработки кадра: np.asanyarray, cvtColor, out.write, imshow)
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            out.write(color_image)
            if show:
                cv2.imshow("RGB Stream", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as e:
        print(
            f"Ошибка времени выполнения: {e}"
        )  # Ловим исключение, что когда файл прерывается и подаем следующий
    except Exception as e:
        print(f"Ошибка: {e}")
        raise e
    finally:
        pipeline.stop()


def main(make_dir=True, eval=False):
    """Главная функция скрипта, выполняющая извлечение RGB-потоков из bag-файлов.

    Args:
        make_dir (bool, optional): Создавать ли необходимые директории. Defaults to True.
        eval (bool, optional): Режим оценки (использует другие bag-файлы). Defaults to False.
    """
    BASE = EVAL if eval else TRAIN

    if make_dir:
        BASE.check_tree()

    for person_class in NAMES:
        class_name = person_class["class"]

        # Обрабатываем все записи (samples) для данного класса
        for sample in person_class["samples"]:
            bn = sample["bag"]  # bag файл
            on = sample["out"]  # выходной файл
            ev = sample["eval"]  # файл для валидации

            if eval and (ev is None or ev == ""):
                continue

            # Определяем пути к bag и mp4 файлам
            if not eval:
                bag_file = BASE.BAG / f"{bn}.bag"
                mp4_file = BASE.MP4 / f"{on}.mp4"
            else:
                bag_file = BASE.BAG / f"{ev}.bag"
                mp4_file = BASE.MP4 / f"{on}.mp4"

            # Проверяем существование mp4 файла, если его нет — извлекаем данные
            if not mp4_file.exists():
                extract_rgb_to_mp4(bag_file, mp4_file)
            else:
                if not mp4_file.is_file():
                    raise RuntimeError(f"Path '{mp4_file=}' exists but is not a file!'")


if __name__ == "__main__":
    main(make_dir=True, eval=False)
