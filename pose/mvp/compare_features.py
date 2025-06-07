# YOU NEED parametrization=False in feature_bake.py TOW MAKE THIS SCRIPT WORK
import os

import matplotlib.pyplot as plt
import numpy as np

from .paths.paths import EVAL, NAMES, PLOTS, TRAIN

eval_dir = EVAL.FEATURES
feature_dir = TRAIN.FEATURES
all_files = sorted([str(feature_dir / file) for file in os.listdir(feature_dir)])
names = {i["bag"]: n for n, i in enumerate(NAMES)}
TRAIN_RATIO = 0.75


# === Загрузка данных ===
def load_and_split_data(file_paths, name_to_label_map):
    train_data, test_data = [], []
    train_labels, test_labels = [], []

    for file in file_paths:
        # Загрузка данных из файла
        arr = np.load(file)

        # Извлечение имени класса из пути файла
        class_name = file[31:-4]
        label = name_to_label_map[class_name]

        # Разделение данных без перемешивания
        split_idx = int(len(arr) * TRAIN_RATIO)

        # Добавление в тренировочные данные
        train_data.append(arr[:split_idx])
        train_labels.append(np.full(split_idx, label))

        # Добавление в тестовые данные
        test_data.append(arr[split_idx:])
        test_labels.append(np.full(len(arr) - split_idx, label))

    return (
        np.vstack(train_data),
        np.concatenate(train_labels),
        np.vstack(test_data),
        np.concatenate(test_labels),
    )


def load_evaluation_data(eval_dir):
    data = []
    file_names = []
    true_labels = []
    for file in sorted(os.listdir(eval_dir)):
        file_path = os.path.join(eval_dir, file)
        arr = np.load(file_path)
        data.append(arr)
        file_names.extend([file] * arr.shape[0])
        class_name = file[:-4]  # Удаляем расширение .npy
        true_labels.extend([names[class_name]] * arr.shape[0])

    val_data = np.vstack(data)
    true_labels = np.array(true_labels)
    return np.vstack(data), file_names, true_labels


def filter_data_by_label(data, labels, target_label):
    labels_np = np.array(labels)
    target_label = np.array(target_label).astype(labels_np.dtype)
    return data[labels_np == target_label]


def get_feature_group(feature_idx):
    """
    Возвращает название группы для признака по индексу.
    """
    if 1 <= feature_idx <= 16:
        return "Углы"
    elif 17 <= feature_idx <= 28:
        return "Длины"
    elif 29 <= feature_idx <= 44:
        return "Производная углов"
    elif 45 <= feature_idx <= 60:
        return "Вторая производная углов"
    elif 61 <= feature_idx <= 72:
        return "Производная длин"
    elif 73 <= feature_idx <= 84:
        return "Вторая производная длин"
    else:
        return "Неизвестная группа"


def plot_features(
    train_data,
    train_labels,
    test_data,
    test_labels,
    val_data,
    val_labels,
    target_name,
    plot_path: str | os.PathLike = PLOTS,
):
    """
    Создает сетку графиков 84x3 (84 признака, 3 набора данных)
    для конкретного имени (target_name) с подписями групп признаков.
    """
    # Получаем метку для целевого имени
    target_label = names[target_name]

    # Фильтруем данные по метке
    train_data_filtered = filter_data_by_label(train_data, train_labels, target_label)
    test_data_filtered = filter_data_by_label(test_data, test_labels, target_label)
    val_data_filtered = filter_data_by_label(val_data, val_labels, target_label)

    # Отладочная информация
    print(f"Размер train_data_filtered: {len(train_data_filtered)}")
    print(f"Размер test_data_filtered: {len(test_data_filtered)}")
    print(f"Размер val_data_filtered: {len(val_data_filtered)}")

    # Проверяем, что данные не пустые
    if (
        len(train_data_filtered) == 0
        or len(test_data_filtered) == 0
        or len(val_data_filtered) == 0
    ):
        print(f"Нет данных для имени: {target_name}")
        print(f"Метка: {target_label}")
        print(f"Уникальные метки в val_labels: {np.unique(val_labels)}")
        return

    plt.figure(figsize=(30, 200))

    # Создаем сетку графиков
    for feature_idx in range(84):
        # Получаем данные для текущего признака
        train_feature = train_data_filtered[:, feature_idx]
        test_feature = test_data_filtered[:, feature_idx]
        val_feature = val_data_filtered[:, feature_idx]

        # Тренировочные данные (первый столбец)
        plt.subplot(84, 3, feature_idx * 3 + 1)
        plt.plot(train_feature, color="blue", alpha=0.7)
        plt.title(
            f"Feature {feature_idx + 1} - Train\n({get_feature_group(feature_idx + 1)})"
        )
        plt.ylabel("Value")
        plt.ylim(train_feature.min(), train_feature.max())
        plt.grid(True)

        # Тестовые данные (второй столбец)
        plt.subplot(84, 3, feature_idx * 3 + 2)
        plt.plot(test_feature, color="green", alpha=0.7)
        plt.title(
            f"Feature {feature_idx + 1} - Test\n({get_feature_group(feature_idx + 1)})"
        )
        plt.ylim(test_feature.min(), test_feature.max())
        plt.grid(True)

        # Валидационные данные (третий столбец)
        plt.subplot(84, 3, feature_idx * 3 + 3)
        plt.plot(val_feature, color="red", alpha=0.7)
        plt.title(
            f"Feature {feature_idx + 1} - Validation\n({get_feature_group(feature_idx + 1)})"
        )
        plt.ylim(val_feature.min(), val_feature.max())
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path / f"all_features_{target_name}.pdf")
    plt.close()


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_and_split_data(
        all_files, names
    )
    val_data, val_files, val_labels = load_evaluation_data(eval_dir)

    target_name = "Oleg_Karasev"  # Имя, для которого нужно построить графики, точно как в names.json
    plot_features(
        train_data,
        train_labels,
        test_data,
        test_labels,
        val_data,
        val_labels,
        target_name,
    )
