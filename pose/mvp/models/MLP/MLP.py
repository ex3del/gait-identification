"""
Модуль для обучения и оценки модели MLP (Multi-Layer Perceptron) для классификации
данных о походке на основе извлеченных признаков.

Скрипт выполняет следующие шаги:
1. Загружает предварительно извлеченные признаки (`.npy` файлы) с помощью функции `load_and_split_data`.
2. Разделяет данные каждого файла на обучающую и тестовую выборки.
3. Выполняет стандартизацию признаков (StandardScaler).
4. Преобразует данные в тензоры PyTorch и создает DataLoader'ы.
5. Определяет архитектуру модели MLP (простую или глубокую).
6. Определяет функцию потерь (CrossEntropy или FocalLoss) и оптимизатор.
7. Запускает цикл обучения модели, включающий:
    - Обучение на обучающей выборке.
    - Оценку на тестовой выборке в конце каждой эпохи.
    - Расчет и логирование метрик (loss, accuracy, F1, precision, recall).
    - Сохранение лучшей модели по точности на тестовой выборке.
    - Генерацию и сохранение графиков метрик, confusion matrix и ROC-кривых.
8. Выполняет финальную оценку лучшей модели на тестовом наборе и выводит classification report.
"""

import os
import pprint
import random
from typing import TYPE_CHECKING, Dict, List, Tuple, Any
from warnings import warn
from pathlib import Path  # Используем pathlib для путей

from joblib import dump, load  # Добавляем load, если нужно будет загружать scaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Импорт путей и функции загрузки данных
from ...paths.paths import TRAIN, EVAL, NAMES, MODELS

from .load import load_and_split_data, TRAIN_RATIO


if TYPE_CHECKING:
    pass


WEIGHTS = MODELS / "MLP" / "MLP_weights"

# --- Генерация словаря ИмяКласса -> Метка ---
# Создаем словарь отображения имен классов из NAMES в целые числа (метки)
# Это используется и в load_and_split_data, и для меток на графиках
try:
    CLASS_NAME_TO_LABEL_MAP: Dict[str, int] = {
        info["class"]: idx for idx, info in enumerate(NAMES) if "class" in info
    }
    if not CLASS_NAME_TO_LABEL_MAP:
        raise ValueError(
            "Структура NAMES не содержит записей с ключом 'class' или пуста."
        )
    # Словарь для обратного отображения (метка -> имя класса), полезно для отчетов
    LABEL_TO_CLASS_NAME_MAP: Dict[int, str] = {
        v: k for k, v in CLASS_NAME_TO_LABEL_MAP.items()
    }
    CLASS_NAMES_ORDERED: List[str] = [
        LABEL_TO_CLASS_NAME_MAP[i] for i in range(len(CLASS_NAME_TO_LABEL_MAP))
    ]
except Exception as e:
    print(f"Ошибка при создании словаря меток из NAMES: {e}")
    print(
        "Структура NAMES должна быть списком словарей, каждый из которых содержит ключ 'class'."
    )
    # Завершаем выполнение, если не можем создать карту меток
    exit(1)

# === Гиперпараметры и настройки ===
BATCH_SIZE = 300000
EPOCHS = 250
LEARNING_RATE = 0.001
HIDDEN_SIZE = 251
HIDDEN_SIZE_2 = 81
USE_DEEP_MODEL = True
DROPOUT_RATE = 0.1
NCOLS = (12 + 12 + 12 + 16 + 16 + 16) * 30  # lengths + angles + angles'
NUM_CLASSES = len(CLASS_NAME_TO_LABEL_MAP)  # Количество людей для классификации
PLOT_METRICS = True
USE_FOCAL_LOSS = False
SAVE_WEIGHTS = True

# Параметры FocalLoss (если USE_FOCAL_LOSS=True)
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0

# Установите seed в начале вашего скрипта или функции main
SEED = 42  # Выберите любое целое число

# Определение устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


# === Определение кастомного датасета ===
class GaitDataset(Dataset):
    """
    Простой класс датасета PyTorch для данных о походке.

    Args:
        X (torch.Tensor): Тензор с признаками.
        y (torch.Tensor): Тензор с метками.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Возвращает общее количество сэмплов в датасете."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает один сэмпл (признаки и метку) по индексу."""
        return self.X[idx], self.y[idx]


# === Определение моделей MLP ===
class MLPClassifier(nn.Module):
    """
    Простая модель MLP с одним скрытым слоем.

    Архитектура: Linear -> Tanh -> Dropout -> Linear (Output)

    Args:
        input_size (int): Размерность входных признаков (NCOLS).
        hidden_size (int): Количество нейронов в скрытом слое.
        num_classes (int): Количество выходных классов.
        dropout_rate (float): Доля dropout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout_rate: float = 0.1,
    ):
        super(MLPClassifier, self).__init__()
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, 1, input_size).

        Returns:
            torch.Tensor: Выходной тензор логитов формы (batch_size, num_classes).
        """
        # x имеет форму (batch_size, 1, input_size) из-за reshape в main
        # Убираем лишнее измерение '1' перед первым линейным слоем
        x = x.squeeze(1)  # Теперь форма (batch_size, input_size)
        out = self.fc0(x)
        out = torch.tanh(out)  # Используем tanh вместо F.tanh для единообразия
        out = self.dropout(out)
        out = self.fc1(out)  # Выходной слой
        return out


class MLPClassifier_deep(nn.Module):
    """
    Более глубокая модель MLP с двумя скрытыми слоями.

    Архитектура: Linear -> Tanh -> Dropout -> Linear -> Tanh -> Dropout -> Linear (Output)

    Args:
        input_size (int): Размерность входных признаков (NCOLS).
        hidden_size (int): Количество нейронов в первом скрытом слое.
        hidden_size_2 (int): Количество нейронов во втором скрытом слое.
        num_classes (int): Количество выходных классов.
        dropout_rate (float): Доля dropout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_size_2: int,
        num_classes: int,
        dropout_rate: float = 0.1,
    ):
        super(MLPClassifier_deep, self).__init__()
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, 1, input_size).

        Returns:
            torch.Tensor: Выходной тензор логитов формы (batch_size, num_classes).
        """
        # x имеет форму (batch_size, 1, input_size) из-за reshape в main
        x = x.squeeze(1)  # Убираем лишнее измерение '1' -> (batch_size, input_size)
        out = self.fc0(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Выходной слой
        return out


# === Определение Focal Loss ===
class FocalLoss(nn.Module):
    """
    Реализация Focal Loss для задач многоклассовой классификации.

    Помогает бороться с дисбалансом классов, фокусируя обучение на "сложных" примерах.

    Args:
        alpha (float or List[float], optional): Веса для балансировки классов.
            Может быть скаляром (применяется ко всем классам) или списком весов для каждого класса.
            Defaults to 1.0 (без балансировки).
        gamma (float, optional): Параметр фокусировки (>= 0). Увеличивает влияние "сложных"
            примеров. Defaults to 2.0.
        reduction (str, optional): Способ агрегации потерь ('none', 'mean', 'sum').
            Defaults to 'mean'.
    """

    def __init__(
        self,
        alpha: float | List[float] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        # Проверка корректности gamma
        if gamma < 0:
            raise ValueError("Параметр gamma должен быть неотрицательным.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет Focal Loss.

        Args:
            inputs (torch.Tensor): Логиты модели (выход до softmax), форма (batch_size, num_classes).
            targets (torch.Tensor): Истинные метки классов, форма (batch_size,).

        Returns:
            torch.Tensor: Скалярное значение потерь (если reduction='mean' или 'sum') или
                          тензор потерь для каждого примера (если reduction='none').
        """
        # Вычисляем log_softmax для численной стабильности
        log_pt = F.log_softmax(inputs, dim=1)
        # Собираем log_pt для истинных классов
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Вычисляем pt (вероятности истинных классов)
        pt = log_pt.exp()

        # Вычисляем основной модулирующий фактор focal loss: (1 - pt)^gamma
        focal_term = (1 - pt).pow(self.gamma)

        # Применяем веса alpha, если они заданы
        if isinstance(self.alpha, (float, int)):
            # Если alpha - скаляр, применяем его ко всем
            alpha_term = self.alpha
        elif isinstance(self.alpha, list) or isinstance(self.alpha, torch.Tensor):
            # Если alpha - список или тензор, выбираем вес для соответствующего класса
            alpha_tensor = torch.tensor(
                self.alpha, device=inputs.device, dtype=torch.float32
            )
            alpha_term = alpha_tensor.gather(0, targets)
        else:
            raise TypeError(
                "Параметр alpha должен быть float, int, list или torch.Tensor"
            )

        # Итоговые потери для каждого примера: alpha * (1 - pt)^gamma * (-log_pt)
        loss = alpha_term * focal_term * (-log_pt)

        # Агрегация потерь
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Неизвестный тип reduction: {self.reduction}")


def set_seed(seed_value=42):
    """Устанавливает зерно для Python, NumPy и PyTorch для CPU и GPU."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # Для GPU (если используется)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # для мульти-GPU
    print(f"Установлен глобальный seed: {seed_value}")


# === Функция Цикла Обучения ===
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,  # Функция потерь (например, CrossEntropyLoss или FocalLoss)
    optimizer: optim.Optimizer,
    epochs: int,
    best_model_path: Path,  # Путь для сохранения лучшей модели
    class_names: List[str],  # Список имен классов для отчетов и графиков
    save_weights: bool = False,  # Флаг сохранения весов
    plot_dir: Path | None = None,  # Директория для сохранения графиков
) -> Dict[str, List[float]]:
    """
    Выполняет цикл обучения и валидации модели.

    Args:
        model (nn.Module): Модель PyTorch для обучения.
        train_loader (DataLoader): DataLoader для обучающих данных.
        test_loader (DataLoader): DataLoader для тестовых данных.
        criterion (nn.Module): Функция потерь.
        optimizer (optim.Optimizer): Оптимизатор.
        epochs (int): Количество эпох обучения.
        best_model_path (Path): Путь для сохранения лучшей модели (.pth файл).
        class_names (List[str]): Список имен классов для использования в отчетах/графиках.
        save_weights (bool, optional): Сохранять ли веса лучшей модели. Defaults to False.
        plot_dir (Path | None, optional): Директория для сохранения графиков метрик.
                                           Если None, графики не сохраняются. Defaults to None.

    Returns:
        Dict[str, List[float]]: Словарь с историями метрик (loss, accuracy, f1, precision, recall)
                                 для обучающей и тестовой выборок по эпохам.
    """
    num_classes = len(class_names)  # Определяем количество классов
    best_test_acc = 0.0  # Лучшая точность на тестовой выборке

    # Списки для хранения истории метрик по эпохам
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "train_f1": [],
        "test_f1": [],
        "train_precision": [],
        "test_precision": [],
        "train_recall": [],
        "test_recall": [],
    }
    # Списки для хранения всех предсказаний и меток за все эпохи (для финальных графиков)
    all_epoch_train_labels = []
    all_epoch_train_preds = []
    all_epoch_test_labels = []
    all_epoch_test_preds = []
    all_epoch_test_probs = []  # Вероятности нужны для ROC-AUC

    print("\n--- Начало обучения ---")
    for epoch in range(epochs):
        # --- Фаза обучения ---
        model.train()  # Переводим модель в режим обучения
        running_train_loss = 0.0
        epoch_train_preds = []
        epoch_train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            optimizer.zero_grad()  # Обнуляем градиенты
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Обновляем веса

            # Сбор статистики для метрик
            running_train_loss += loss.item() * inputs.size(0)  # Умножаем на batch_size
            _, preds = torch.max(outputs, 1)  # Получаем предсказанный класс
            epoch_train_preds.extend(preds.cpu().numpy())
            epoch_train_labels.extend(labels.cpu().numpy())

        # Расчет метрик для обучающей выборки за эпоху
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(epoch_train_labels, epoch_train_preds)
        # Используем 'weighted' для учета дисбаланса классов, zero_division=0 для избежания предупреждений
        epoch_train_f1 = f1_score(
            epoch_train_labels, epoch_train_preds, average="weighted", zero_division=0
        )
        epoch_train_precision = precision_score(
            epoch_train_labels, epoch_train_preds, average="weighted", zero_division=0
        )
        epoch_train_recall = recall_score(
            epoch_train_labels, epoch_train_preds, average="weighted", zero_division=0
        )

        # Сохранение метрик эпохи
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["train_f1"].append(epoch_train_f1)
        history["train_precision"].append(epoch_train_precision)
        history["train_recall"].append(epoch_train_recall)
        # Сохранение предсказаний/меток для финальных графиков (только последняя эпоха)
        if epoch == epochs - 1:
            all_epoch_train_labels = epoch_train_labels
            all_epoch_train_preds = epoch_train_preds

        # --- Фаза оценки (валидации) ---
        model.eval()  # Переводим модель в режим оценки
        running_test_loss = 0.0
        epoch_test_preds = []
        epoch_test_labels = []
        epoch_test_probs = []  # Вероятности для ROC-AUC

        with torch.no_grad():  # Отключаем вычисление градиентов
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs, dim=1)  # Получаем вероятности
                _, preds = torch.max(outputs, 1)

                epoch_test_preds.extend(preds.cpu().numpy())
                epoch_test_labels.extend(labels.cpu().numpy())
                epoch_test_probs.extend(probs.cpu().numpy())  # Сохраняем вероятности

        # Расчет метрик для тестовой выборки за эпоху
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_acc = accuracy_score(epoch_test_labels, epoch_test_preds)
        epoch_test_f1 = f1_score(
            epoch_test_labels, epoch_test_preds, average="weighted", zero_division=0
        )
        epoch_test_precision = precision_score(
            epoch_test_labels, epoch_test_preds, average="weighted", zero_division=0
        )
        epoch_test_recall = recall_score(
            epoch_test_labels, epoch_test_preds, average="weighted", zero_division=0
        )

        # Сохранение метрик эпохи
        history["test_loss"].append(epoch_test_loss)
        history["test_acc"].append(epoch_test_acc)
        history["test_f1"].append(epoch_test_f1)
        history["test_precision"].append(epoch_test_precision)
        history["test_recall"].append(epoch_test_recall)
        # Сохранение предсказаний/меток/вероятностей для финальных графиков (только последняя эпоха)
        if epoch == epochs - 1:
            all_epoch_test_labels = epoch_test_labels
            all_epoch_test_preds = epoch_test_preds
            all_epoch_test_probs = epoch_test_probs

        # Вывод прогресса
        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"  Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | F1: {epoch_train_f1:.4f}"
        )
        print(
            f"  Test Loss:  {epoch_test_loss:.4f} | Acc: {epoch_test_acc:.4f} | F1: {epoch_test_f1:.4f}"
        )
        # print(f"  Test Precision: {epoch_test_precision:.4f} | Recall: {epoch_test_recall:.4f}")
        print("-" * 50)

        # Сохранение лучшей модели
        if save_weights and epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            try:
                # Убедимся, что директория существует
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"  >> Лучшая модель сохранена в {best_model_path} (Test Acc: {best_test_acc:.4f})"
                )
            except Exception as e:
                warn(f"Не удалось сохранить модель: {e}")

    print("--- Обучение завершено ---")

    # --- Построение и сохранение графиков (если указано) ---
    if plot_dir and PLOT_METRICS:
        print(f"Сохранение графиков в {plot_dir}...")
        try:
            plot_dir.mkdir(parents=True, exist_ok=True)
            epochs_range = range(1, epochs + 1)

            plt.style.use("seaborn-v0_8-darkgrid")  # Используем стиль seaborn
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))  # 3 строки, 2 столбца
            fig.suptitle("Метрики Обучения и Валидации", fontsize=16)

            # График 1: Loss
            axes[0, 0].plot(
                epochs_range, history["train_loss"], "b-o", label="Train Loss"
            )
            axes[0, 0].plot(
                epochs_range, history["test_loss"], "r-o", label="Test Loss"
            )
            axes[0, 0].set_title("Функция Потерь")
            axes[0, 0].set_xlabel("Эпоха")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # График 2: Accuracy
            axes[0, 1].plot(
                epochs_range, history["train_acc"], "b-o", label="Train Accuracy"
            )
            axes[0, 1].plot(
                epochs_range, history["test_acc"], "r-o", label="Test Accuracy"
            )
            axes[0, 1].set_title("Точность (Accuracy)")
            axes[0, 1].set_xlabel("Эпоха")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # График 3: F1-Score
            axes[1, 0].plot(
                epochs_range, history["train_f1"], "b-o", label="Train F1 (Weighted)"
            )
            axes[1, 0].plot(
                epochs_range, history["test_f1"], "r-o", label="Test F1 (Weighted)"
            )
            axes[1, 0].set_title("F1-Score (Weighted)")
            axes[1, 0].set_xlabel("Эпоха")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # График 4: Precision & Recall (можно объединить или выбрать одно)
            axes[1, 1].plot(
                epochs_range,
                history["test_precision"],
                "g-o",
                label="Test Precision (Weighted)",
            )
            axes[1, 1].plot(
                epochs_range,
                history["test_recall"],
                "m-o",
                label="Test Recall (Weighted)",
            )
            axes[1, 1].set_title("Precision & Recall (Weighted) - Test")
            axes[1, 1].set_xlabel("Эпоха")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            # График 5: Confusion Matrix (используем данные последней эпохи)
            ax_cm = axes[2, 0]
            if all_epoch_test_labels and all_epoch_test_preds:
                cm = confusion_matrix(
                    all_epoch_test_labels,
                    all_epoch_test_preds,
                    labels=range(num_classes),
                )
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=class_names
                )
                disp.plot(ax=ax_cm, cmap=plt.cm.Blues, xticks_rotation=45)
                ax_cm.set_title("Confusion Matrix (Test Set - Last Epoch)")
            else:
                ax_cm.text(
                    0.5,
                    0.5,
                    "Нет данных для CM",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax_cm.set_title("Confusion Matrix")

            # График 6: ROC Curves (One-vs-Rest) (используем данные последней эпохи)
            ax_roc = axes[2, 1]
            if all_epoch_test_labels and all_epoch_test_probs and num_classes > 1:
                try:
                    y_test_bin = label_binarize(
                        np.array(all_epoch_test_labels), classes=range(num_classes)
                    )
                    all_probs_np = np.array(all_epoch_test_probs)

                    if y_test_bin.shape[1] == 1 and num_classes == 2:  # Бинарный случай
                        fpr, tpr, _ = roc_curve(
                            y_test_bin[:, 0], all_probs_np[:, 1]
                        )  # Используем вероятности класса 1
                        roc_auc = auc(fpr, tpr)
                        ax_roc.plot(
                            fpr, tpr, label=f"Class 1 vs Rest (AUC = {roc_auc:.3f})"
                        )
                    else:  # Многоклассовый случай
                        for i in range(num_classes):
                            if (
                                np.sum(y_test_bin[:, i]) > 0
                            ):  # Проверяем, что класс есть в данных
                                fpr, tpr, _ = roc_curve(
                                    y_test_bin[:, i], all_probs_np[:, i]
                                )
                                roc_auc = auc(fpr, tpr)
                                # Обрезаем имя класса для легенды, если слишком длинное
                                class_label = (
                                    class_names[i][:15] + "..."
                                    if len(class_names[i]) > 15
                                    else class_names[i]
                                )
                                ax_roc.plot(
                                    fpr,
                                    tpr,
                                    label=f"{class_label} (AUC = {roc_auc:.3f})",
                                )

                    ax_roc.plot([0, 1], [0, 1], "k--", label="Случайное угадывание")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title("ROC кривые (One-vs-Rest)")
                    ax_roc.legend(fontsize="small")  # Уменьшаем шрифт легенды
                    ax_roc.grid(True)

                except Exception as roc_e:
                    ax_roc.text(
                        0.5,
                        0.5,
                        f"Ошибка ROC: {roc_e}",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                    ax_roc.set_title("ROC кривые (Ошибка)")
            else:
                ax_roc.text(
                    0.5,
                    0.5,
                    "Нет данных/классов для ROC",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax_roc.set_title("ROC кривые")

            plt.tight_layout(
                rect=[0, 0.03, 1, 0.97]
            )  # Корректируем отступы из-за suptitle
            plot_path = plot_dir / "mlp_training_metrics.png"
            plt.savefig(plot_path)
            plt.close(fig)  # Закрываем фигуру, чтобы освободить память
            print(f"Графики сохранены в {plot_path}")

        except Exception as plot_e:
            warn(f"Не удалось построить или сохранить графики: {plot_e}")

    return history


# === Функция Оценки Модели ===
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],  # Имена классов для classification_report
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет оценку обученной модели на тестовом наборе данных.

    Args:
        model (nn.Module): Обученная модель PyTorch.
        test_loader (DataLoader): DataLoader для тестовых данных.
        class_names (List[str]): Список имен классов для отчета.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Кортеж из двух NumPy массивов:
            1. Истинные метки (all_labels).
            2. Предсказанные метки (all_preds).
    """
    print("\n--- Оценка лучшей модели на тестовом наборе ---")
    model.eval()  # Переводим модель в режим оценки
    all_preds = []
    all_labels = []
    with torch.no_grad():  # Отключаем градиенты
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Преобразуем списки в NumPy массивы
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # Вывод базовой точности
    acc = accuracy_score(all_labels_np, all_preds_np)
    print(f"Итоговая точность на тесте: {acc:.4f}")

    # Вывод полного отчета о классификации
    print("\nОтчет о классификации:")
    # Получаем уникальные метки, присутствующие в данных, для labels
    present_labels = np.unique(np.concatenate((all_labels_np, all_preds_np)))
    # Создаем target_names только для присутствующих меток
    present_class_names = [class_names[i] for i in present_labels]

    # Выводим отчет только для присутствующих классов
    try:
        report = classification_report(
            all_labels_np,
            all_preds_np,
            labels=present_labels,
            target_names=present_class_names,
            zero_division=0,
        )
        print(report)
    except Exception as report_e:
        print(f"Не удалось сгенерировать classification_report: {report_e}")
        # Попытка вывести отчет без имен, если проблема в них
        try:
            report = classification_report(all_labels_np, all_preds_np, zero_division=0)
            print("Отчет без имен классов:\n", report)
        except Exception as simple_report_e:
            print(f"Не удалось сгенерировать даже простой отчет: {simple_report_e}")

    return all_labels_np, all_preds_np


# === Основная функция запуска ===
def main(
    scaler_path: Path = WEIGHTS / "LSTM_train_scaler.joblib",
    best_weights_path: Path = WEIGHTS / "best_LSTM.pth",
    plot_dir_path: Path = WEIGHTS,
    use_deep: bool = USE_DEEP_MODEL,
    epochs_override: int | None = None,  # Возможность переопределить EPOCHS
):
    """
    Главная функция, оркестрирующая процесс обучения и оценки MLP модели.

    Args:
        scaler_path (Path, optional): Путь для сохранения/загрузки объекта StandardScaler.
        best_weights_path (Path, optional): Путь для сохранения лучшей модели.
        plot_dir_path (Path, optional): Директория для сохранения графиков.
        use_deep (bool, optional): Использовать ли глубокую MLP модель.
        epochs_override (int | None, optional): Позволяет переопределить количество эпох
                                                из глобальных настроек. Defaults to None.
    """
    assert WEIGHTS.exists(), str(WEIGHTS)
    print("--- Запуск пайплайна обучения MLP ---")
    # Задаем Random seed для воспроизводимости
    set_seed(SEED)

    # === 1. Загрузка и разделение данных ===
    try:
        X_train, y_train, X_test, y_test = load_and_split_data(CLASS_NAME_TO_LABEL_MAP)
    except KeyError as e:
        print(f"Критическая ошибка: Класс из NAMES не найден в словаре меток. {e}")
        return
    except FileNotFoundError as e:
        print(f"Критическая ошибка: Не найден файл признаков при загрузке. {e}")
        return
    except ValueError as e:
        print(
            f"Критическая ошибка: Проблема с данными при загрузке или объединении. {e}"
        )
        return
    except Exception as e:
        print(f"Критическая ошибка при загрузке данных: {e}")
        import traceback

        traceback.print_exc()
        return

    # Проверка, что данные загружены
    if X_train.size == 0 or X_test.size == 0:
        print("Ошибка: Обучающие или тестовые данные пусты после загрузки. Прерывание.")
        return

    # Определяем количество признаков (NCOLS) из загруженных данных
    if X_train.ndim > 1:
        NCOLS = X_train.shape[1]
    elif X_train.ndim == 1:  # Если признаки 1-мерные
        NCOLS = 1
        # Может потребоваться reshape для StandardScaler и модели
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        warn(
            "Входные данные 1-мерные. Выполнено reshape(-1, 1). Убедитесь, что модель ожидает такой формат."
        )
    else:
        print("Ошибка: Не удалось определить количество признаков (NCOLS).")
        return

    if NCOLS <= 0:
        print("Ошибка: Количество признаков (NCOLS) равно нулю или отрицательно.")
        return

    print(f"Определено количество признаков (NCOLS): {NCOLS}")

    # === 2. Предобработка данных ===
    print("\n--- Предобработка данных (StandardScaler) ---")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Сохранение скейлера
        dump(scaler, scaler_path)
        print(f"StandardScaler обучен и сохранен в {scaler_path}")
    except Exception as e:
        print(f"Ошибка при масштабировании данных или сохранении скейлера: {e}")
        return

    # === 3. Преобразование в тензоры PyTorch и создание DataLoader'ов ===
    print("\n--- Создание тензоров и DataLoader'ов ---")
    try:
        # Преобразование в тензоры PyTorch
        # Добавляем измерение '1' посередине
        # Форма станет (batch_size, 1, NCOLS).
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).reshape(
            -1, 1, NCOLS
        )
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).reshape(
            -1, 1, NCOLS
        )
        y_train_tensor = torch.tensor(
            y_train, dtype=torch.long
        )  # Метки для CrossEntropy должны быть Long
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Создание датасетов
        train_dataset = GaitDataset(X_train_tensor, y_train_tensor)
        test_dataset = GaitDataset(X_test_tensor, y_test_tensor)

        # Создание DataLoader'ов
        # shuffle=True для обучающего загрузчика обычно рекомендуется
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"DataLoader'ы созданы (Batch size: {BATCH_SIZE})")
        print(f"Количество батчей в train_loader: {len(train_loader)}")
        print(f"Количество батчей в test_loader: {len(test_loader)}")

    except Exception as e:
        print(f"Ошибка при создании тензоров или DataLoader'ов: {e}")
        return

    # === 4. Инициализация модели ===
    print("\n--- Инициализация модели MLP ---")
    if use_deep:
        print("Используется MLPClassifier_deep")
        model = MLPClassifier_deep(
            input_size=NCOLS,
            hidden_size=HIDDEN_SIZE,
            hidden_size_2=HIDDEN_SIZE_2,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        ).to(device)
    else:
        print("Используется MLPClassifier")
        model = MLPClassifier(
            input_size=NCOLS,
            hidden_size=HIDDEN_SIZE,  # Используем HIDDEN_SIZE вместо 99
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        ).to(device)
    print(model)  # Печатаем архитектуру модели

    # === 5. Определение функции потерь и оптимизатора ===
    if USE_FOCAL_LOSS:
        print(f"Используется FocalLoss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean"
        ).to(device)
    else:
        print("Используется CrossEntropyLoss")
        # nn.CrossEntropyLoss включает в себя LogSoftmax и NLLLoss
        criterion = nn.CrossEntropyLoss().to(device)

    # Оптимизатор Adam с L2 регуляризацией (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    print(f"Оптимизатор: Adam (LR={LEARNING_RATE}, weight_decay=0.0001)")

    # === 6. Обучение модели ===
    epochs_to_run = epochs_override if epochs_override is not None else EPOCHS
    print(f"Количество эпох: {epochs_to_run}")

    try:
        train_history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs_to_run,
            best_model_path=best_weights_path,
            class_names=CLASS_NAMES_ORDERED,  # Передаем упорядоченный список имен
            save_weights=SAVE_WEIGHTS,
            plot_dir=plot_dir_path if PLOT_METRICS else None,
        )
    except Exception as e:
        print(f"Ошибка во время обучения модели: {e}")
        import traceback

        traceback.print_exc()
        return

    # === 7. Финальная оценка модели ===
    # Загрузка лучшей модели (если она сохранялась) для финальной оценки
    if SAVE_WEIGHTS and best_weights_path.exists():
        print(f"\nЗагрузка лучшей модели из {best_weights_path} для финальной оценки.")
        try:
            # Нужно пересоздать модель той же архитектуры перед загрузкой state_dict
            if use_deep:
                final_model = MLPClassifier_deep(
                    NCOLS, HIDDEN_SIZE, HIDDEN_SIZE_2, NUM_CLASSES, DROPOUT_RATE
                ).to(device)
            else:
                final_model = MLPClassifier(
                    NCOLS, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_RATE
                ).to(device)
            final_model.load_state_dict(
                torch.load(best_weights_path, map_location=device)
            )
            evaluate_model(final_model, test_loader, CLASS_NAMES_ORDERED)
        except Exception as e:
            print(f"Ошибка при загрузке лучшей модели или финальной оценке: {e}")
            print("Оценка будет проведена на модели из последней эпохи.")
            # Оцениваем модель из последней эпохи, если не удалось загрузить лучшую
            evaluate_model(model, test_loader, CLASS_NAMES_ORDERED)
    else:
        print(
            "\nФинальная оценка модели из последней эпохи (лучшая модель не сохранялась или не найдена)."
        )
        evaluate_model(model, test_loader, CLASS_NAMES_ORDERED)

    print("\n--- Пайплайн обучения MLP завершен ---")


# --- Точка входа ---
if __name__ == "__main__":
    # Запуск основной функции с путями по умолчанию
    main()
