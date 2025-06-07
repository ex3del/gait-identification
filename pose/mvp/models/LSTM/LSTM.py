"""
Скрипт для обучения и оценки модели LSTM для классификации походки
на основе временных последовательностей извлеченных признаков.

Рабочий процесс:
1.  **Импорт зависимостей:** Загрузка необходимых библиотек (PyTorch, NumPy, Scikit-learn, Matplotlib, etc.)
    и пользовательских модулей (paths, data_loader.load).
2.  **Конфигурация:** Определение глобальных путей, гиперпараметров модели (длина последовательности,
    размер скрытого слоя, количество слоев), параметров обучения (размер батча, эпохи, скорость обучения),
    и выбор устройства (CPU/GPU).
3.  **Загрузка и подготовка данных:**
    -   Использование функции `create_sequences_from_files` из модуля `load` для:
        -   Загрузки файлов `.npy`, содержащих покадровые признаки.
        -   Создания перекрывающихся временных последовательностей (`SEQUENCE_LENGTH`)
          с заданным шагом (`STRIDE`).
        -   Разделения сгенерированных последовательностей из *каждого* файла на
          обучающую и тестовую выборки (`TRAIN_RATIO`).
    -   Масштабирование признаков с помощью `StandardScaler`, обученного *только*
      на кадрах из обучающих последовательностей. Scaler сохраняется для
      последующего использования.
4.  **Создание Dataset и DataLoader:**
    -   Определение класса `GaitSequenceDataset` для обертки списков тензоров
      последовательностей и их меток.
    -   Создание экземпляров `DataLoader` для эффективной подачи данных в модель
      батчами, с перемешиванием для обучающей выборки.
5.  **Определение Модели:**
    -   Класс `GaitClassifierLSTM`, реализующий многослойную (возможно, двунаправленную)
      LSTM сеть, за которой следуют слои Dropout и полносвязный слой (Linear)
      для финальной классификации.
6.  **Функция Потерь и Оптимизатор:**
    -   Определение класса `FocalLoss` (альтернатива для несбалансированных данных).
    -   Использование стандартной `nn.CrossEntropyLoss` для мультиклассовой классификации.
    -   Инициализация оптимизатора `AdamW` с заданными параметрами.
7.  **Обучение и Оценка:**
    -   Функция `train_model` реализует цикл обучения по эпохам:
        -   Фаза обучения: прогон по `train_loader`, вычисление потерь, обратное
          распространение ошибки, обновление весов.
        -   Фаза оценки: прогон по `test_loader` (без вычисления градиентов),
          расчет метрик на тестовой выборке.
        -   Расчет и сохранение метрик (loss, accuracy, F1, precision, recall)
          для обеих фаз на каждой эпохе.
        -   Сохранение весов модели, показавшей лучшую производительность
          (наименьший test loss) на тестовой выборке.
        -   Генерация и сохранение комплексного графика с кривыми обучения,
          матрицей ошибок, ROC-кривыми, распределением классов и отчетом
          классификации (по результатам последней эпохи).
    -   Функция `evaluate_model` для финальной оценки производительности
      загруженной лучшей модели на тестовом наборе.
8.  **Основная функция `main`:**
    -   Оркестрирует весь процесс: вызов загрузки данных, масштабирования,
      создания загрузчиков, инициализации модели/оптимизатора, запуска
      обучения и финальной оценки.

      Воспроизводимость:
- Устанавливаются seed'ы для `random`, `numpy`, `torch` (CPU и CUDA).
- Используется детерминированный режим для cuDNN (`torch.backends.cudnn.deterministic`).
- Отключается бенчмаркинг cuDNN (`torch.backends.cudnn.benchmark`), который может вносить вариативность.
- Для `DataLoader` используется `worker_init_fn` для корректной инициализации seed'ов
  в дочерних процессах (`numpy`, `random`) и `torch.Generator` для контроля
  над порядком сэмплирования (перемешивания).
- **Примечание:** Воспроизводимость гарантируется при одинаковых:
    - Версиях библиотек (Python, PyTorch, NumPy, CUDA, cuDNN).
    - Оборудовании (особенно GPU).
    - Конфигурации скрипта (включая `num_workers` в DataLoader).

"""

import os
import random
import pprint
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
from joblib import dump, load
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, TYPE_CHECKING
from warnings import warn
import traceback

from ...paths.paths import TRAIN, EVAL, NAMES, MODELS
from .load import (
    create_sequences_from_files,
    CLASS_NAME_TO_LABEL_MAP,
    LABEL_TO_CLASS_NAME_MAP,
    CLASS_NAMES_ORDERED,
    NUM_CLASSES,
    TRAIN_RATIO,  # Импортируем, если используется только здесь
)

# --- Импорт pandas опционально для графиков ---
try:
    import pandas as pd
except ImportError:
    pd = None
    warn(
        "Библиотека pandas не установлена. График распределения классов не будет построен."
    )

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Union


WEIGHTS = MODELS / "LSTM" / "LSTM_weights"


# **Настройки Воспроизводимости**
RANDOM_SEED: int = 42  # Фиксированный seed для всех источников случайности

# --- Гиперпараметры Модели и Обучения ---

# Параметры данных
SEQUENCE_LENGTH: int = 30  # Длина каждой входной последовательности (количество кадров)
STRIDE: int = 5  # Шаг, с которым создаются последовательности из кадров файла
INPUT_SIZE_PER_FRAME: int = 84  # Количество признаков, описывающих ОДИН кадр (ВАЖНО!)

# Параметры LSTM Модели
HIDDEN_SIZE: int = 186  # Размер скрытого состояния LSTM (подбирается экспериментально)
NUM_LAYERS: int = 4  # Количество слоев LSTM (глубина)
USE_BIDIRECTIONAL: bool = True  # Использовать ли двунаправленную LSTM (True/False)

# Параметры FFN головы
USE_FFN_HEAD: bool = False  # Использовать ли FFN голову (True/False)
FFN_HIDDEN_SIZE: int = 128  # Размер скрытого слоя в FFN голове
FFN_DROPOUT: float = 0.6  # Dropout внутри FFN головы

# Параметры Обучения
# NUM_CLASSES уже импортирован из data_loader
BATCH_SIZE: int = 32  # Количество последовательностей в одном батче (зависит от GPU)
EPOCHS: int = 15  # Количество полных проходов по обучающему набору данных
LEARNING_RATE: float = 1e-4  # Скорость обучения оптимизатора
WEIGHT_DECAY: float = (
    0.1  # Коэффициент L2 регуляризации для оптимизатора AdamW (помогает бороться с переобучением)
)
# TRAIN_RATIO импортирован из data_loader (доля последовательностей из файла для обучения)

# Параметры Dropout (для регуляризации)
LSTM_DROPOUT: float = (
    0.6 if NUM_LAYERS > 1 else 0.0
)  # Dropout между слоями LSTM (только если слоев > 1)
FC_DROPOUT: float = 0.4  # Dropout перед последним полносвязным слоем


# --- Определение устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Функции для Воспроизводимости ===


def set_seed(seed: int):
    """
    Устанавливает seed для Python, NumPy и PyTorch для обеспечения воспроизводимости.
    Также настраивает детерминированное поведение cuDNN.

    Args:
        seed (int): Значение seed для инициализации генераторов случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # для multi-GPU
        # Настройки для детерминированного поведения cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Настройки cuDNN для воспроизводимости включены.")
    else:
        print("CUDA недоступна, настройки cuDNN пропущены.")
    print(f"[ Установлен Seed : {seed} ]")


def seed_worker(worker_id: int):
    """
    Функция инициализации для воркеров DataLoader.

    Устанавливает уникальный, но детерминированный seed для NumPy и random
    в каждом воркере на основе глобального seed PyTorch и ID воркера.
    PyTorch сам устанавливает свой seed для каждого воркера.

    Args:
        worker_id (int): ID текущего воркера DataLoader.
    """
    # Получаем seed, который PyTorch установил для этого воркера
    # и берем по модулю 2**32 для предотвращения переполнения
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print(f"Воркер {worker_id} инициализирован с seed: {worker_seed}")  # Для отладки


# === Кастомный датасет ===
class GaitSequenceDataset(Dataset):
    """
    Датасет PyTorch для последовательностей признаков походки.

    Оборачивает список тензоров PyTorch, где каждый тензор представляет собой
    одну временную последовательность признаков формы [sequence_length, input_size_per_frame],
    и соответствующий список целочисленных меток классов.
    """

    def __init__(self, sequences: List[torch.Tensor], labels: List[int]):
        """
        Инициализация датасета.

        Args:
            sequences (List[torch.Tensor]): Список тензоров, где каждый тензор - одна
                                           последовательность признаков.
            labels (List[int]): Список целочисленных меток, соответствующий каждой
                                последовательности.

        Raises:
            ValueError: Если списки пусты или имеют разную длину.
        """
        if not sequences or not labels or len(sequences) != len(labels):
            raise ValueError(
                "Списки последовательностей и меток не должны быть пустыми и должны иметь одинаковую длину."
            )
        self.sequences = sequences
        self.labels = labels
        seq_shape = sequences[0].shape if sequences else "(пусто)"
        print(
            f"Создан датасет с {len(self.sequences)} последовательностями. Пример формы: {seq_shape}"
        )

    def __len__(self) -> int:
        """Возвращает общее количество последовательностей в датасете."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Возвращает одну последовательность и соответствующую ей метку по индексу.

        Args:
            idx (int): Индекс запрашиваемого элемента.

        Returns:
            Tuple[torch.Tensor, int]: Кортеж (тензор_последовательности, метка_класса).
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label


# === Обновленная Модель LSTM с FFN головой ===
class GaitClassifierLSTM(nn.Module):
    """
    Модель LSTM для классификации временных последовательностей признаков походки.

    Архитектура состоит из одного или нескольких слоев LSTM (возможно, двунаправленных),
    за которыми следует опциональная двухслойная FFN голова (MLP) и финальный
    полносвязный слой для получения логитов классов.
    """

    def __init__(
        self,
        input_size: int,  # Признаков на кадр
        hidden_size: int,  # Размер LSTM
        num_layers: int,  # Слоев LSTM
        num_classes: int,  # Количество классов
        use_bidirectional: bool = USE_BIDIRECTIONAL,  # Двунаправленная LSTM?
        lstm_dropout: float = LSTM_DROPOUT,  # Dropout между слоями LSTM
        use_ffn_head: bool = USE_FFN_HEAD,  # Использовать FFN голову?
        ffn_hidden_size: int = FFN_HIDDEN_SIZE,  # Скрытый размер FFN
        ffn_dropout: float = FFN_DROPOUT,  # Dropout в FFN
    ):
        """
        Инициализация слоев модели.

        Args:
            input_size (int): Количество признаков в одном кадре.
            hidden_size (int): Размер скрытого состояния LSTM.
            num_layers (int): Количество рекуррентных слоев LSTM.
            num_classes (int): Количество выходных классов.
            use_bidirectional (bool): Использовать ли двунаправленную LSTM.
            lstm_dropout (float): Вероятность Dropout между слоями LSTM (если num_layers > 1).
            use_ffn_head (bool): Добавлять ли двухслойную FFN голову перед финальной классификацией.
            ffn_hidden_size (int): Размер скрытого слоя в FFN голове.
            ffn_dropout (float): Вероятность Dropout внутри FFN головы (после активации).
        """
        super(GaitClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if use_bidirectional else 1
        self.use_ffn_head = use_ffn_head

        # --- Слой LSTM ---
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=use_bidirectional,
            dropout=lstm_dropout,
        )

        # --- Определяем входной размер для классификационной головы ---
        classifier_input_size = hidden_size * self.num_directions

        # --- Классификационная Голова (Либо FFN, либо один Linear слой) ---
        if self.use_ffn_head:
            print(
                f"Используется FFN голова с размером {ffn_hidden_size} и dropout {ffn_dropout}"
            )
            self.classifier_head = nn.Sequential(
                nn.Linear(
                    classifier_input_size, ffn_hidden_size
                ),  # Первый линейный слой FFN
                nn.ReLU(),  # Функция активации (можно GELU или др.)
                nn.Dropout(ffn_dropout),  # Dropout внутри FFN
                nn.Linear(ffn_hidden_size, num_classes),  # Второй (выходной) слой FFN
            )
        else:
            print("Используется один линейный слой для классификации.")
            # Если FFN не используется, нужен Dropout перед финальным слоем
            self.dropout_fc = nn.Dropout(
                ffn_dropout
            )  # Используем ffn_dropout для консистентности
            self.classifier_head = nn.Linear(
                classifier_input_size, num_classes
            )  # Один финальный слой

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Определяет прямой проход данных через модель.

        Args:
            x (torch.Tensor): Входной тензор с последовательностями признаков.
                              Форма: (batch_size, sequence_length, input_size_per_frame).

        Returns:
            torch.Tensor: Выходной тензор с логитами для каждого класса.
                          Форма: (batch_size, num_classes).
        """
        batch_size = x.size(0)

        # Инициализация начальных состояний LSTM
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(x.device)

        # Пропускаем данные через LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Используем выход LSTM только с последнего временного шага
        last_step_out = out[
            :, -1, :
        ]  # Форма: (batch_size, hidden_size * num_directions)

        # Пропускаем через классификационную голову
        if self.use_ffn_head:
            # FFN голова уже включает все необходимые слои (Linear, ReLU, Dropout, Linear)
            logits = self.classifier_head(last_step_out)
        else:
            # Применяем Dropout и затем один линейный слой
            out_dropout = self.dropout_fc(last_step_out)
            logits = self.classifier_head(out_dropout)

        # Форма logits: (batch_size, num_classes)
        return logits


# === Focal Loss ===
class FocalLoss(nn.Module):
    """
    Реализация Focal Loss для решения проблемы дисбаланса классов в задачах классификации.
    Уменьшает вклад легко классифицируемых примеров во время обучения.

    Args:
        alpha (float or list or torch.Tensor): Весовой коэффициент для классов.
            Может быть скаляром (применяется ко всем классам одинаково) или
            списком/тензором весов для каждого класса. Помогает бороться с дисбалансом.
        gamma (float): Параметр фокусировки (>= 0). Увеличивает относительный вес
            трудно классифицируемых примеров (с низкой предсказанной вероятностью
            правильного класса). При gamma=0 превращается в стандартную CrossEntropy (с alpha).
        reduction (str): Способ агрегации потерь по батчу: 'none', 'mean', 'sum'.
                         'mean': усреднить потери по батчу.
                         'sum': суммировать потери по батчу.
                         'none': вернуть потери для каждого примера отдельно.
    """

    def __init__(
        self,
        alpha: Union[float, list, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        # Валидация параметров
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Недопустимое значение reduction: {reduction}. Допустимы 'mean', 'sum', 'none'."
            )
        if gamma < 0:
            raise ValueError("Параметр gamma должен быть неотрицательным.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет Focal Loss.

        Args:
            inputs (torch.Tensor): Логиты модели (выход до softmax). Форма: (N, C), где N - размер батча, C - кол-во классов.
            targets (torch.Tensor): Истинные метки классов. Форма: (N,), значения от 0 до C-1.

        Returns:
            torch.Tensor: Вычисленное значение Focal Loss (скаляр, если reduction='mean'/'sum', или тензор формы (N,), если reduction='none').
        """
        num_classes = inputs.size(1)

        # Вычисляем log softmax для численной стабильности
        log_probs = F.log_softmax(inputs, dim=1)
        # Получаем вероятности классов
        probs = torch.exp(log_probs)

        # Создаем one-hot encoding для целевых меток
        # Форма target_one_hot: (N, C)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Выбираем log_probs для истинных классов
        # log_probs.gather(1, targets.unsqueeze(1)) вернет (N, 1), убираем последнюю размерность
        log_probs_true_class = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Выбираем вероятности для истинных классов
        probs_true_class = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Вычисляем модулирующий фактор (1 - p_t)^gamma
        focal_weights = torch.pow(1 - probs_true_class, self.gamma)

        # Обрабатываем alpha
        if isinstance(self.alpha, (list, tuple)):
            # Преобразуем список в тензор один раз, если нужно
            if (
                not hasattr(self, "alpha_tensor")
                or self.alpha_tensor.device != inputs.device
            ):
                self.alpha_tensor = torch.tensor(
                    self.alpha, device=inputs.device, dtype=torch.float32
                )
            # Собираем веса alpha для каждого примера в батче
            alpha_weights = self.alpha_tensor.gather(0, targets)
        elif isinstance(self.alpha, torch.Tensor):
            # Если alpha уже тензор, просто выбираем нужные веса
            alpha_weights = self.alpha.to(inputs.device).gather(0, targets)
        elif isinstance(self.alpha, (int, float)):
            # Если alpha - скаляр, он применяется ко всем одинаково
            alpha_weights = self.alpha
        else:
            raise TypeError("Параметр alpha должен быть float, list или torch.Tensor.")

        # Вычисляем итоговые потери для каждого примера: -alpha * (1 - p_t)^gamma * log(p_t)
        loss_per_sample = -alpha_weights * focal_weights * log_probs_true_class

        # Применяем редукцию
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample


# === Цикл обучения ===
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    best_model_path: Path,
    save_weights: bool = True,
    plot_dir: Path = None,
    class_names_ordered: List[str] = None,
) -> Dict[str, List[float]]:
    """
    Обучает и оценивает модель LSTM на протяжении заданного числа эпох.

    В каждой эпохе выполняет:
    1. Фазу обучения на `train_loader`.
    2. Фазу оценки на `test_loader`.
    3. Рассчитывает и логирует метрики (loss, accuracy, F1, precision, recall).
    4. Сохраняет лучшую модель (по минимальному test loss) если `save_weights=True`.
    5. В конце обучения строит и сохраняет графики метрик, если указан `plot_dir`.

    Args:
        model (nn.Module): Модель PyTorch для обучения.
        train_loader (DataLoader): Загрузчик данных для обучения.
        test_loader (DataLoader): Загрузчик данных для тестирования/валидации.
        criterion (nn.Module): Функция потерь (например, CrossEntropyLoss, FocalLoss).
        optimizer (optim.Optimizer): Оптимизатор (например, AdamW).
        epochs (int): Общее количество эпох обучения.
        best_model_path (Path): Путь для сохранения файла с весами лучшей модели.
        save_weights (bool): Флаг, указывающий, нужно ли сохранять лучшую модель.
        plot_dir (Path, optional): Директория для сохранения графиков обучения.
                                   Если None, графики не сохраняются.
        class_names_ordered (List[str], optional): Упорядоченный список имен классов
                                                  для подписей на графиках.

    Returns:
        Dict[str, List[float]]: Словарь, содержащий списки метрик для каждой эпохи
                                (например, 'train_loss', 'test_acc', etc.).
    """
    print("\n--- Начало обучения ---")
    best_loss = float("inf")  # Инициализируем лучшую потерю бесконечностью

    # Словарь для хранения истории метрик по эпохам
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
    # Словарь для хранения данных последней эпохи (для графиков)
    final_epoch_data = {
        "train_labels": [],
        "train_preds": [],
        "train_probs": [],
        "test_labels": [],
        "test_preds": [],
        "test_probs": [],
    }

    # Основной цикл по эпохам
    for epoch in range(epochs):
        # --- Фаза обучения ---
        model.train()  # Переводим модель в режим обучения (активирует Dropout и т.д.)
        running_loss = 0.0
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_probs = []

        # Итерация по батчам из обучающего загрузчика
        for i, (sequences, labels) in enumerate(train_loader):
            # Перемещаем данные на выбранное устройство (CPU/GPU)
            sequences, labels = sequences.to(device), labels.to(device)

            # Обнуляем градиенты перед новым вычислением
            optimizer.zero_grad()

            # Прямой проход: получаем выходы модели (логиты)
            outputs = model(sequences)

            # Вычисляем значение функции потерь
            loss = criterion(outputs, labels)

            # Обратное распространение ошибки: вычисляем градиенты
            loss.backward()

            # Шаг оптимизатора: обновляем веса модели
            optimizer.step()

            # Накапливаем потери для статистики по эпохе
            running_loss += loss.item()

            # Получаем вероятности и предсказанные классы для расчета метрик
            with torch.no_grad():  # Не считаем градиенты для этой части
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                epoch_train_preds.extend(preds.cpu().numpy())
                epoch_train_labels.extend(labels.cpu().numpy())
                epoch_train_probs.extend(probs.cpu().numpy())  # Сохраняем вероятности

        # Расчет средних метрик для обучающей выборки за эпоху
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(epoch_train_labels, epoch_train_preds)
        # average='weighted' учитывает дисбаланс классов при расчете F1, Precision, Recall
        epoch_f1 = f1_score(
            epoch_train_labels, epoch_train_preds, average="weighted", zero_division=0
        )
        epoch_precision = precision_score(
            epoch_train_labels, epoch_train_preds, average="weighted", zero_division=0
        )
        epoch_recall = recall_score(
            epoch_train_labels, epoch_train_preds, average="weighted", zero_division=0
        )

        # Сохраняем метрики в историю
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["train_f1"].append(epoch_f1)
        history["train_precision"].append(epoch_precision)
        history["train_recall"].append(epoch_recall)

        # Если это последняя эпоха, сохраняем детальные данные для графиков
        if epoch == epochs - 1:
            final_epoch_data["train_labels"] = epoch_train_labels
            final_epoch_data["train_preds"] = epoch_train_preds
            final_epoch_data["train_probs"] = epoch_train_probs

        # --- Фаза Оценки ---
        model.eval()  # Переводим модель в режим оценки (отключает Dropout)
        test_loss = 0.0
        epoch_test_preds = []
        epoch_test_labels = []
        epoch_test_probs = []

        # Отключаем вычисление градиентов для фазы оценки
        with torch.no_grad():
            # Итерация по батчам из тестового загрузчика
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                # Прямой проход
                outputs = model(sequences)

                # Вычисление потерь
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Получаем вероятности и предсказания
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                epoch_test_preds.extend(preds.cpu().numpy())
                epoch_test_labels.extend(labels.cpu().numpy())
                epoch_test_probs.extend(probs.cpu().numpy())  # Сохраняем вероятности

        # Расчет средних метрик для тестовой выборки за эпоху
        test_epoch_loss = test_loss / len(test_loader)
        test_epoch_acc = accuracy_score(epoch_test_labels, epoch_test_preds)
        test_epoch_f1 = f1_score(
            epoch_test_labels, epoch_test_preds, average="weighted", zero_division=0
        )
        test_epoch_precision = precision_score(
            epoch_test_labels, epoch_test_preds, average="weighted", zero_division=0
        )
        test_epoch_recall = recall_score(
            epoch_test_labels, epoch_test_preds, average="weighted", zero_division=0
        )

        # Сохраняем метрики в историю
        history["test_loss"].append(test_epoch_loss)
        history["test_acc"].append(test_epoch_acc)
        history["test_f1"].append(test_epoch_f1)
        history["test_precision"].append(test_epoch_precision)
        history["test_recall"].append(test_epoch_recall)

        # Если это последняя эпоха, сохраняем детальные данные для графиков
        if epoch == epochs - 1:
            final_epoch_data["test_labels"] = epoch_test_labels
            final_epoch_data["test_preds"] = epoch_test_preds
            final_epoch_data["test_probs"] = epoch_test_probs

        # --- Логирование результатов эпохи ---
        print(f"--- Эпоха {epoch+1}/{epochs} ---")
        print(
            f"Обучение | Потери: {epoch_loss:.4f} | Точность: {epoch_acc:.4f} | F1: {epoch_f1:.4f}"
        )
        print(
            f"Тест     | Потери: {test_epoch_loss:.4f} | Точность: {test_epoch_acc:.4f} | F1: {test_epoch_f1:.4f}"
        )
        print("-" * 70)

        # --- Сохранение лучшей модели ---
        # Сохраняем модель, если текущая тестовая потеря лучше предыдущей лучшей
        if save_weights and test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            try:
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"*** Лучшая модель сохранена (Эпоха {epoch+1}) с Test Loss: {best_loss:.4f} по пути: {best_model_path} ***"
                )
            except Exception as e:
                warn(f"Не удалось сохранить модель: {e}")

    print("--- Обучение завершено ---")

    # --- Построение и сохранение графиков (если указана директория) ---
    if plot_dir:
        print(f"\nПостроение и сохранение графиков в {plot_dir}...")
        plot_dir.mkdir(exist_ok=True, parents=True)
        try:
            plt.style.use("seaborn-v0_8-darkgrid")  # Стиль графиков
            # Создаем фигуру с сеткой 3x3 для 9 графиков
            fig, axes = plt.subplots(3, 3, figsize=(20, 18))  # Размер фигуры
            axes = axes.flatten()  # Преобразуем массив осей в одномерный для удобства

            # 1-5: Кривые обучения для основных метрик
            metrics_to_plot = ["loss", "acc", "f1", "precision", "recall"]
            titles = [
                "Потери",
                "Точность",
                "F1-мера",
                "Точность (Precision)",
                "Полнота (Recall)",
            ]
            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i]
                # Отображаем кривые для train и test, добавляем последнее значение в легенду
                ax.plot(
                    history[f"train_{metric}"],
                    label=f"Обучение ({history[f'train_{metric}'][-1]:.3f})",
                    marker=".",
                )
                ax.plot(
                    history[f"test_{metric}"],
                    label=f"Тест ({history[f'test_{metric}'][-1]:.3f})",
                    marker=".",
                )
                ax.set_title(f"Кривые обучения: {titles[i]}")
                ax.set_xlabel("Эпохи")
                ax.set_ylabel(titles[i])
                ax.legend()
                ax.grid(True)

            # 6: Матрица ошибок (Confusion Matrix)
            ax = axes[5]
            cm = confusion_matrix(
                final_epoch_data["test_labels"], final_epoch_data["test_preds"]
            )
            # Обрезаем имена классов для лучшей читаемости на осях
            display_labels = (
                [name[:12] for name in class_names_ordered]
                if class_names_ordered
                else None
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=display_labels
            )
            disp.plot(
                ax=ax, cmap="Blues", xticks_rotation="vertical"
            )  # Вертикальные подписи по оси X
            ax.set_title("Матрица ошибок (Тест, последняя эпоха)")
            # Уменьшаем шрифт чисел в ячейках и на осях, если классов много
            if NUM_CLASSES > 15:
                for labels_cm in ax.texts:
                    labels_cm.set_fontsize(8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

            # 7: ROC-кривые (One-vs-Rest)
            ax = axes[6]
            test_labels_np = np.array(final_epoch_data["test_labels"])
            test_probs_np = np.array(final_epoch_data["test_probs"])
            # Бинаризация меток для подхода One-vs-Rest
            y_test_bin = label_binarize(test_labels_np, classes=np.arange(NUM_CLASSES))
            if NUM_CLASSES > 1:  # ROC имеет смысл для > 1 класса
                fpr, tpr, roc_auc_dict = dict(), dict(), dict()
                # Строим ROC-кривую для каждого класса против всех остальных
                for i in range(NUM_CLASSES):
                    # Проверяем, есть ли в тестовой выборке примеры обоих типов (класс i и не класс i)
                    if len(np.unique(y_test_bin[:, i])) > 1:
                        fpr[i], tpr[i], _ = roc_curve(
                            y_test_bin[:, i], test_probs_np[:, i]
                        )
                        roc_auc_dict[i] = auc(fpr[i], tpr[i])
                        class_name = (
                            class_names_ordered[i][:12]
                            if class_names_ordered
                            else str(i)
                        )
                        ax.plot(
                            fpr[i],
                            tpr[i],
                            lw=2,
                            label=f"{class_name} (AUC={roc_auc_dict[i]:.2f})",
                        )
                    else:
                        warn(
                            f"Класс {i} ('{class_names_ordered[i]}') не имеет примеров обоих типов в тесте, ROC-кривая не строится."
                        )
                # Линия случайного угадывания
                ax.plot([0, 1], [0, 1], "k--", label="Случайное угадывание")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("Доля ложноположительных срабатываний (FPR)")
                ax.set_ylabel("Доля истинноположительных срабатываний (TPR)")
                ax.set_title("ROC-кривые (One-vs-Rest)")
                ax.legend(
                    loc="lower right", fontsize=8 if NUM_CLASSES > 10 else None
                )  # Уменьшаем шрифт легенды
                ax.grid(True)
            else:
                ax.text(
                    0.5, 0.5, "ROC не строится для 1 класса", ha="center", va="center"
                )

            # 8: Распределение классов в Train/Test
            ax = axes[7]
            if pd:  # Строим, только если pandas доступен
                train_labels_np = np.array(final_epoch_data["train_labels"])
                test_labels_np = np.array(final_epoch_data["test_labels"])
                # Создаем DataFrame для удобства с seaborn
                df_train = pd.DataFrame({"label": train_labels_np, "split": "Обучение"})
                df_test = pd.DataFrame({"label": test_labels_np, "split": "Тест"})
                df_combined = pd.concat([df_train, df_test])
                # Получаем подписи для оси X
                tick_labels = (
                    [name[:12] for name in class_names_ordered]
                    if class_names_ordered
                    else None
                )
                # Строим гистограмму
                sns.histplot(
                    data=df_combined,
                    x="label",
                    hue="split",
                    bins=NUM_CLASSES,
                    discrete=True,
                    multiple="dodge",
                    shrink=0.8,
                    ax=ax,
                )
                ax.set_title("Распределение классов (посчитано по посл. эпохе)")
                ax.set_xlabel("Класс")
                ax.set_ylabel("Количество последовательностей")
                ax.set_xticks(np.arange(NUM_CLASSES))
                if tick_labels:
                    ax.set_xticklabels(
                        tick_labels,
                        rotation=90,
                        fontsize=8 if NUM_CLASSES > 15 else None,
                    )
                ax.grid(axis="y")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Pandas не установлен,\nграфик распределения\nклассов не построен.",
                    ha="center",
                    va="center",
                    color="grey",
                )
                ax.set_title("Распределение классов")

            # 9: Текстовый отчет Classification Report
            ax = axes[8]
            ax.axis("off")  # Отключаем оси координат
            try:
                # Используем те же метки, что и для confusion matrix/ROC
                report_labels = np.arange(NUM_CLASSES)
                report_target_names = (
                    [name[:20] for name in class_names_ordered]
                    if class_names_ordered
                    else None
                )  # Имена подлиннее для отчета
                report = classification_report(
                    final_epoch_data["test_labels"],
                    final_epoch_data["test_preds"],
                    labels=report_labels,
                    target_names=report_target_names,
                    zero_division=0,
                    digits=3,  # Количество знаков после запятой
                )
                # Выводим текст отчета на оси
                ax.text(
                    0.01,
                    0.99,
                    report,
                    family="monospace",
                    va="top",
                    ha="left",
                    fontsize=7,
                )  # Моноширинный шрифт
                ax.set_title(
                    "Classification Report (Тест, последняя эпоха)", fontsize=10
                )
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Не удалось создать отчет:\n{e}",
                    ha="center",
                    va="center",
                    color="red",
                )

            # Общая настройка и сохранение фигуры
            plt.tight_layout(
                rect=[0, 0.03, 1, 0.97]
            )  # Добавляем отступ для общего заголовка
            fig.suptitle(
                f"Результаты обучения LSTM (Эпох: {EPOCHS}, SeqLen: {SEQUENCE_LENGTH}, Hidden: {HIDDEN_SIZE})",
                fontsize=16,
            )
            save_path = plot_dir / "lstm_training_metrics.png"
            plt.savefig(save_path, dpi=150)  # Сохраняем с повышенным разрешением
            plt.close(fig)  # Закрываем фигуру, чтобы освободить память
            print(f"Графики сохранены в {save_path}")
        except Exception as e:
            warn(f"Не удалось построить или сохранить графики: {e}")
            traceback.print_exc()  # Печатаем стектрейс ошибки

    return history


# === Функция оценки ===
def evaluate_model(
    model: nn.Module, test_loader: DataLoader
) -> Tuple[List[int], List[int]]:
    """
    Оценивает производительность обученной модели на тестовом наборе данных.

    Переводит модель в режим оценки, проходит по всем тестовым данным,
    собирает истинные и предсказанные метки, вычисляет и печатает
    итоговую точность (accuracy).

    Args:
        model (nn.Module): Обученная модель PyTorch.
        test_loader (DataLoader): Загрузчик данных для тестирования.

    Returns:
        Tuple[List[int], List[int]]: Кортеж, содержащий два списка:
                                     1. Список истинных меток.
                                     2. Список предсказанных меток.
    """
    print("\n--- Финальная оценка модели на тестовых данных ---")
    model.eval()  # Переключаем модель в режим оценки
    all_preds = []
    all_labels = []
    # Отключаем вычисление градиентов
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            # Прямой проход
            outputs = model(sequences)
            # Получение предсказаний (индекс класса с максимальной вероятностью)
            _, preds = torch.max(outputs, 1)
            # Сохраняем предсказания и истинные метки
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Расчет и вывод итоговой точности
    acc = accuracy_score(all_labels, all_preds)
    print(f"Итоговая точность на тесте: {acc:.4f}")
    return all_labels, all_preds


# === Основная функция ===
def main(
    scaler_path: Path = WEIGHTS / "lstm_scaler.joblib",
    best_weights_path: Path = WEIGHTS / "best_lstm_model.pth",
    plot_save_dir: Path = WEIGHTS,
):
    """
    Главная функция скрипта.

    Оркестрирует весь процесс:
    1. Загрузка и создание последовательностей данных.
    2. Масштабирование признаков.
    3. Создание DataLoader'ов.
    4. Инициализация модели LSTM, функции потерь и оптимизатора.
    5. Запуск цикла обучения и оценки `train_model`.
    6. Загрузка лучшей сохраненной модели и вывод финального отчета `classification_report`.

    Args:
        scaler_path (Path): Путь для сохранения/загрузки объекта StandardScaler.
        best_weights_path (Path): Путь для сохранения/загрузки весов лучшей модели.
        plot_save_dir (Path): Директория для сохранения графиков обучения.
    """
    print("--- Запуск основного скрипта обучения LSTM ---")

    print(f"--- Параметры Запуска ---")
    print(f"Используемое устройство: {device}")
    print(f"Количество классов: {NUM_CLASSES}")
    print(f"Длина последовательности (кадров): {SEQUENCE_LENGTH}")
    print(f"Шаг создания последовательностей (stride): {STRIDE}")
    print(f"Признаков на кадр: {INPUT_SIZE_PER_FRAME}")
    print(f"Размер скрытого слоя LSTM: {HIDDEN_SIZE}")
    print(f"Количество слоев LSTM: {NUM_LAYERS}")
    print(f"Использовать Bidirectional LSTM: {USE_BIDIRECTIONAL}")
    print(f"Размер батча: {BATCH_SIZE}")
    print(f"Количество эпох: {EPOCHS}")
    print(f"Скорость обучения: {LEARNING_RATE}")
    print(f"Weight Decay (AdamW): {WEIGHT_DECAY}")
    print(f"Dropout LSTM: {LSTM_DROPOUT}")
    print(f"Dropout FC: {FC_DROPOUT}")
    print(f"Соотношение Train/Test из файла: {TRAIN_RATIO*100:.1f}%")
    print(f"Директория сохранения модели: {WEIGHTS}")
    print("-" * 25)

    # !!! 0. УСТАНОВКА SEED ДЛЯ ВОСПРОИЗВОДИМОСТИ !!!
    set_seed(RANDOM_SEED)

    # 1. Создание последовательностей данных
    try:
        # Вызов функции из модуля load.py
        train_seqs, train_lbls, test_seqs, test_lbls = create_sequences_from_files(
            feature_dir=TRAIN.FEATURES,
            names_structure=NAMES,
            class_map=CLASS_NAME_TO_LABEL_MAP,
            stride=STRIDE,  # Передаем шаг
            seq_length=SEQUENCE_LENGTH,
            train_ratio=TRAIN_RATIO,
            input_size_per_frame=INPUT_SIZE_PER_FRAME,  # Для проверки размерности
        )
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при создании последовательностей:")
        traceback.print_exc()
        return  # Прерываем выполнение

    # 2. Предобработка данных (Масштабирование)
    print("\n--- Предобработка данных (Масштабирование) ---")
    try:
        num_train_sequences = len(train_seqs)
        if num_train_sequences == 0:
            raise ValueError("Нет обучающих последовательностей для обучения scaler.")

        # Объединяем ВСЕ кадры из ВСЕХ обучающих последовательностей в один большой 2D массив
        # для корректного обучения StandardScaler.
        # Форма до cat: List[Tensor(seq_len, features)] -> После cat: Tensor(N*seq_len, features)
        all_train_frames = torch.cat(train_seqs, dim=0).numpy()
        print(
            f"Форма объединенных обучающих кадров для scaler: {all_train_frames.shape}"
        )

        # Создаем и обучаем Scaler ТОЛЬКО на обучающих данных
        scaler = StandardScaler()
        scaler.fit(all_train_frames)

        # Сохраняем обученный Scaler
        dump(scaler, scaler_path)
        print(f"Scaler обучен и сохранен в {scaler_path}")

        # Применяем обученный Scaler к КАЖДОЙ последовательности (train и test)
        # Преобразуем тензор -> numpy -> transform -> tensor
        scaled_train_seqs = [
            torch.tensor(scaler.transform(seq.numpy()), dtype=torch.float32)
            for seq in train_seqs
        ]
        scaled_test_seqs = [
            torch.tensor(scaler.transform(seq.numpy()), dtype=torch.float32)
            for seq in test_seqs
        ]
        print("Масштабирование train и test последовательностей завершено.")
        # Очистка памяти (опционально)
        del all_train_frames, train_seqs, test_seqs

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при масштабировании данных:")
        traceback.print_exc()
        return

    # 3. Создание Датасетов и DataLoader'ов
    print("\n--- Создание Датасетов и DataLoader'ов ---")
    try:
        # Создаем объекты Dataset
        train_dataset = GaitSequenceDataset(scaled_train_seqs, train_lbls)
        test_dataset = GaitSequenceDataset(scaled_test_seqs, test_lbls)
        # Очистка памяти (опционально)
        del scaled_train_seqs, train_lbls, scaled_test_seqs, test_lbls

        # !!! Создаем генератор для DataLoader и устанавливаем seed !!!
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)

        # Создаем объекты DataLoader
        # shuffle=True для обучающего загрузчика - важно для стохастичности обучения
        # num_workers > 0 ускоряет загрузку данных в несколько потоков
        # pin_memory=True ускоряет передачу данных на GPU (если используется)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=device.type == "cuda",
            generator=g,
        )
        # shuffle=False для тестового загрузчика - порядок не важен, но важна воспроизводимость
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=device.type == "cuda",
        )
        print(f"DataLoader'ы созданы. Размер батча: {BATCH_SIZE}")
        print(f"Количество батчей (train): {len(train_loader)}")
        print(f"Количество батчей (test): {len(test_loader)}")

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при создании Dataset/DataLoader:")
        traceback.print_exc()
        return

    # 4. Инициализация модели, функции потерь и оптимизатора
    print("\n--- Инициализация модели, потерь и оптимизатора ---")
    try:
        # Создаем экземпляр модели LSTM
        model = GaitClassifierLSTM(
            input_size=INPUT_SIZE_PER_FRAME,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            use_bidirectional=USE_BIDIRECTIONAL,  # Передаем параметр
            lstm_dropout=LSTM_DROPOUT,  # Передаем параметр
            use_ffn_head=USE_FFN_HEAD,  # Передаем флаг FFN
            ffn_hidden_size=FFN_HIDDEN_SIZE,  # Передаем размер FFN
            ffn_dropout=FFN_DROPOUT,  # Передаем dropout FFN
        ).to(
            device
        )  # Перемещаем модель на выбранное устройство
        print("Структура модели:")
        print(model)

        # Выбор функции потерь
        # criterion = FocalLoss(alpha=0.75, gamma=2.0).to(device) # Пример с FocalLoss
        criterion = nn.CrossEntropyLoss().to(
            device
        )  # Стандартный выбор для мультиклассовой классификации
        print(f"Функция потерь: {criterion.__class__.__name__}")

        # Выбор оптимизатора
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        print(
            f"Оптимизатор: {optimizer.__class__.__name__} (LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY})"
        )

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при инициализации модели/потерь/оптимизатора:")
        traceback.print_exc()
        return

    # 5. Обучение модели
    try:
        # Запускаем основную функцию обучения
        training_history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            best_model_path=best_weights_path,  # Путь для сохранения лучшей модели
            save_weights=True,  # Флаг сохранения
            plot_dir=plot_save_dir,  # Директория для графиков
            class_names_ordered=CLASS_NAMES_ORDERED,  # Имена классов для графиков
        )
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА во время обучения модели:")
        traceback.print_exc()
        return

    # 6. Финальная оценка лучшей модели
    print("\n--- Загрузка и финальная оценка лучшей модели ---")
    try:
        # Создаем чистую инстанцию модели той же архитектуры
        best_model = GaitClassifierLSTM(
            input_size=INPUT_SIZE_PER_FRAME,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            use_bidirectional=USE_BIDIRECTIONAL,  # Передаем параметр
            lstm_dropout=LSTM_DROPOUT,  # Передаем параметр
            use_ffn_head=USE_FFN_HEAD,  # Передаем флаг FFN
            ffn_hidden_size=FFN_HIDDEN_SIZE,  # Передаем размер FFN
            ffn_dropout=FFN_DROPOUT,  # Передаем dropout FFN
        ).to(device)
        # Загружаем сохраненные веса лучшей модели
        best_model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"Веса лучшей модели загружены из {best_weights_path}")

        # Запускаем оценку на тестовом наборе
        final_labels, final_preds = evaluate_model(best_model, test_loader)

        # Вывод финального отчета classification_report
        print("\n--- Финальный отчет по классификации (лучшая модель) ---")
        # Получаем имена классов для отчета, обрезаем длинные
        report_target_names = (
            [name[:25] for name in CLASS_NAMES_ORDERED] if CLASS_NAMES_ORDERED else None
        )
        report = classification_report(
            final_labels,
            final_preds,
            labels=np.arange(NUM_CLASSES),  # Указываем метки от 0 до N-1
            target_names=report_target_names,  # Передаем имена
            zero_division=0,  # Как обрабатывать деление на ноль (0 или 'warn')
            digits=3,  # Количество знаков после запятой в отчете
        )
        print(report)

    except FileNotFoundError:
        print(
            f"Ошибка: Не удалось найти файл с весами лучшей модели: {best_weights_path}"
        )
        print("Финальный отчет не может быть создан.")
    except Exception as e:
        print(f"\nОшибка при финальной оценке лучшей модели:")
        traceback.print_exc()

    print("\n--- Скрипт успешно завершен ---")


# --- Точка входа в скрипт ---
if __name__ == "__main__":
    # Определяем пути к файлам scaler и весов модели
    scaler_save_path = WEIGHTS / "lstm_scaler.joblib"
    best_model_save_path = WEIGHTS / "best_lstm_model.pth"
    # Запускаем основную функцию main
    main(
        scaler_path=scaler_save_path,
        best_weights_path=best_model_save_path,
        plot_save_dir=WEIGHTS,  # Графики сохраняем в ту же директорию
    )
