# utils/data_utils.py или в вашем скрипте
from dvc.repo import Repo
from pathlib import Path


def setup_data():
    """
    Загружает данные через DVC автоматически.
    """
    print("🔄 Проверка и загрузка данных через DVC...")

    # Список всех .dvc файлов для загрузки
    dvc_targets = []

    # Проверяем какие .dvc файлы есть
    possible_targets = [
        "data/rsvid/train",
        "data/rsvid/eval",
        "data/rsvid/train/kp_3d",
        "data/rsvid/eval/kp_3d",
    ]

    for target in possible_targets:
        dvc_file = Path(f"{target}.dvc")
        if dvc_file.exists():
            dvc_targets.append(target)
            print(f"Найден {dvc_file}")

    # Загружаем все найденные цели через Repo API
    if dvc_targets:
        print(f"Загружаем данные через DVC: {dvc_targets}")
        try:
            # Используем Repo класс вместо dvc.api.pull
            repo = Repo(".")
            repo.pull(targets=dvc_targets)
            print("✅ Данные успешно загружены через DVC")
        except Exception as e:
            print(f"❌ Ошибка при загрузке данных через DVC: {e}")
            # Попробуем загрузить каждую цель отдельно
            for target in dvc_targets:
                try:
                    print(f"Пытаемся загрузить {target} отдельно...")
                    repo.pull(targets=[target])
                    print(f"✅ {target} успешно загружен")
                except Exception as target_error:
                    print(f"❌ Ошибка при загрузке {target}: {target_error}")
    else:
        print("⚠️ Не найдено .dvc файлов для загрузки")

    # Создаем папки features локально
    features_dirs = [
        Path("data/rsvid/train/features"),
        Path("data/rsvid/eval/features"),
    ]

    for features_dir in features_dirs:
        features_dir.mkdir(parents=True, exist_ok=True)

    print("✅ Настройка данных завершена")


if __name__ == "__main__":
    setup_data()
