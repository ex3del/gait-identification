"""
Скрипт для обучения и оценки модели LSTM для классификации походки
на основе временных последовательностей извлеченных признаков.

Теперь все параметры управляются через Hydra конфигурацию.
"""

import random
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union
from warnings import warn

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset

from ...paths.paths import MODELS, NAMES, TRAIN
from .load import (
    CLASS_NAME_TO_LABEL_MAP,
    CLASS_NAMES_ORDERED,
    NUM_CLASSES,
    create_sequences_from_files,
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

    Args:
        worker_id (int): ID текущего воркера DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# === Кастомный датасет ===
class GaitSequenceDataset(Dataset):
    """
    Датасет PyTorch для последовательностей признаков походки.
    """

    def __init__(self, sequences: List[torch.Tensor], labels: List[int]):
        """
        Инициализация датасета.

        Args:
            sequences (List[torch.Tensor]): Список тензоров последовательностей.
            labels (List[int]): Список целочисленных меток.
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
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label


# === Модель LSTM с FFN головой ===
class GaitClassifierLSTM(nn.Module):
    """
    Модель LSTM для классификации временных последовательностей признаков походки.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        use_bidirectional: bool = True,
        lstm_dropout: float = 0.6,
        use_ffn_head: bool = False,
        ffn_hidden_size: int = 128,
        ffn_dropout: float = 0.6,
    ):
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
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        # --- Определяем входной размер для классификационной головы ---
        classifier_input_size = hidden_size * self.num_directions

        # --- Классификационная Голова ---
        if self.use_ffn_head:
            print(
                f"Используется FFN голова с размером {ffn_hidden_size} и dropout {ffn_dropout}"
            )
            self.classifier_head = nn.Sequential(
                nn.Linear(classifier_input_size, ffn_hidden_size),
                nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_hidden_size, num_classes),
            )
        else:
            print("Используется один линейный слой для классификации.")
            self.dropout_fc = nn.Dropout(ffn_dropout)
            self.classifier_head = nn.Linear(classifier_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        last_step_out = out[:, -1, :]

        # Пропускаем через классификационную голову
        if self.use_ffn_head:
            logits = self.classifier_head(last_step_out)
        else:
            out_dropout = self.dropout_fc(last_step_out)
            logits = self.classifier_head(out_dropout)

        return logits


# === Focal Loss ===
class FocalLoss(nn.Module):
    """
    Реализация Focal Loss для решения проблемы дисбаланса классов.
    """

    def __init__(
        self,
        alpha: Union[float, list, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
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
        num_classes = inputs.size(1)

        # Вычисляем log softmax для численной стабильности
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Создаем one-hot encoding для целевых меток
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Выбираем log_probs для истинных классов
        log_probs_true_class = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probs_true_class = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Вычисляем модулирующий фактор (1 - p_t)^gamma
        focal_weights = torch.pow(1 - probs_true_class, self.gamma)

        # Обрабатываем alpha
        if isinstance(self.alpha, (list, tuple)):
            if (
                not hasattr(self, "alpha_tensor")
                or self.alpha_tensor.device != inputs.device
            ):
                self.alpha_tensor = torch.tensor(
                    self.alpha, device=inputs.device, dtype=torch.float32
                )
            alpha_weights = self.alpha_tensor.gather(0, targets)
        elif isinstance(self.alpha, torch.Tensor):
            alpha_weights = self.alpha.to(inputs.device).gather(0, targets)
        elif isinstance(self.alpha, (int, float)):
            alpha_weights = self.alpha
        else:
            raise TypeError("Параметр alpha должен быть float, list или torch.Tensor.")

        # Вычисляем итоговые потери
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
    cfg: DictConfig,
    device: torch.device,
    best_model_path: Path,
    class_names_ordered: List[str] = None,
) -> Dict[str, List[float]]:
    """
    Обучает и оценивает модель LSTM на протяжении заданного числа эпох.

    Args:
        model (nn.Module): Модель PyTorch для обучения.
        train_loader (DataLoader): Загрузчик данных для обучения.
        test_loader (DataLoader): Загрузчик данных для тестирования.
        criterion (nn.Module): Функция потерь.
        optimizer (optim.Optimizer): Оптимизатор.
        cfg (DictConfig): Конфигурация Hydra.
        device (torch.device): Устройство для вычислений.
        best_model_path (Path): Путь для сохранения лучшей модели.
        class_names_ordered (List[str], optional): Имена классов для графиков.

    Returns:
        Dict[str, List[float]]: История метрик обучения.
    """
    print("\n--- Начало обучения ---")
    best_loss = float("inf")
    epochs = cfg.training.training.epochs

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
        model.train()
        running_loss = 0.0
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_probs = []

        # Итерация по батчам из обучающего загрузчика
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Получаем вероятности и предсказанные классы
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                epoch_train_preds.extend(preds.cpu().numpy())
                epoch_train_labels.extend(labels.cpu().numpy())
                epoch_train_probs.extend(probs.cpu().numpy())

        # Расчет средних метрик для обучающей выборки за эпоху
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(epoch_train_labels, epoch_train_preds)
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

        # Если это последняя эпоха, сохраняем детальные данные
        if epoch == epochs - 1:
            final_epoch_data["train_labels"] = epoch_train_labels
            final_epoch_data["train_preds"] = epoch_train_preds
            final_epoch_data["train_probs"] = epoch_train_probs

        # --- Фаза Оценки ---
        model.eval()
        test_loss = 0.0
        epoch_test_preds = []
        epoch_test_labels = []
        epoch_test_probs = []

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                epoch_test_preds.extend(preds.cpu().numpy())
                epoch_test_labels.extend(labels.cpu().numpy())
                epoch_test_probs.extend(probs.cpu().numpy())

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

        # Если это последняя эпоха, сохраняем детальные данные
        if epoch == epochs - 1:
            final_epoch_data["test_labels"] = epoch_test_labels
            final_epoch_data["test_preds"] = epoch_test_preds
            final_epoch_data["test_probs"] = epoch_test_probs

        # --- Логирование результатов эпохи ---
        if cfg.training.logging.log_every_epoch:
            print(f"--- Эпоха {epoch+1}/{epochs} ---")
            print(
                f"Обучение | Потери: {epoch_loss:.4f} | Точность: {epoch_acc:.4f} | F1: {epoch_f1:.4f}"
            )
            print(
                f"Тест     | Потери: {test_epoch_loss:.4f} | Точность: {test_epoch_acc:.4f} | F1: {test_epoch_f1:.4f}"
            )
            print("-" * 70)

        # --- Сохранение лучшей модели ---
        if cfg.training.saving.save_weights and test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            try:
                torch.save(model.state_dict(), best_model_path)
                if cfg.training.logging.verbose:
                    print(
                        f"*** Лучшая модель сохранена (Эпоха {epoch+1}) с Test Loss: {best_loss:.4f} ***"
                    )
            except Exception as e:
                warn(f"Не удалось сохранить модель: {e}")

    print("--- Обучение завершено ---")

    # --- Построение графиков ---
    if cfg.training.saving.save_plots:
        plot_dir = Path(cfg.data.paths.plots_dir) / cfg.training.saving.plots_dirname
        print(f"\nПостроение и сохранение графиков в {plot_dir}...")
        save_training_plots(
            history, final_epoch_data, plot_dir, cfg, class_names_ordered
        )

    return history


def save_training_plots(
    history: Dict[str, List[float]],
    final_epoch_data: Dict[str, List],
    plot_dir: Path,
    cfg: DictConfig,
    class_names_ordered: List[str] = None,
):
    """
    Создает и сохраняет графики обучения.
    """
    plot_dir.mkdir(exist_ok=True, parents=True)
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        axes = axes.flatten()

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

        # 6: Матрица ошибок
        ax = axes[5]
        cm = confusion_matrix(
            final_epoch_data["test_labels"], final_epoch_data["test_preds"]
        )
        display_labels = (
            [name[:12] for name in class_names_ordered] if class_names_ordered else None
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
        ax.set_title("Матрица ошибок (Тест, последняя эпоха)")

        # 7: ROC-кривые
        ax = axes[6]
        test_labels_np = np.array(final_epoch_data["test_labels"])
        test_probs_np = np.array(final_epoch_data["test_probs"])
        y_test_bin = label_binarize(test_labels_np, classes=np.arange(NUM_CLASSES))

        if NUM_CLASSES > 1:
            fpr, tpr, roc_auc_dict = dict(), dict(), dict()
            for i in range(NUM_CLASSES):
                if len(np.unique(y_test_bin[:, i])) > 1:
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_probs_np[:, i])
                    roc_auc_dict[i] = auc(fpr[i], tpr[i])
                    class_name = (
                        class_names_ordered[i][:12] if class_names_ordered else str(i)
                    )
                    ax.plot(
                        fpr[i],
                        tpr[i],
                        lw=2,
                        label=f"{class_name} (AUC={roc_auc_dict[i]:.2f})",
                    )
            ax.plot([0, 1], [0, 1], "k--", label="Случайное угадывание")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Доля ложноположительных срабатываний (FPR)")
            ax.set_ylabel("Доля истинноположительных срабатываний (TPR)")
            ax.set_title("ROC-кривые (One-vs-Rest)")
            ax.legend(loc="lower right", fontsize=8 if NUM_CLASSES > 10 else None)
            ax.grid(True)

        # 8: Распределение классов
        ax = axes[7]
        if pd:
            train_labels_np = np.array(final_epoch_data["train_labels"])
            test_labels_np = np.array(final_epoch_data["test_labels"])
            df_train = pd.DataFrame({"label": train_labels_np, "split": "Обучение"})
            df_test = pd.DataFrame({"label": test_labels_np, "split": "Тест"})
            df_combined = pd.concat([df_train, df_test])

            tick_labels = (
                [name[:12] for name in class_names_ordered]
                if class_names_ordered
                else None
            )
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
            ax.set_title("Распределение классов")
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

        # 9: Classification Report
        ax = axes[8]
        ax.axis("off")
        try:
            report_labels = np.arange(NUM_CLASSES)
            report_target_names = (
                [name[:20] for name in class_names_ordered]
                if class_names_ordered
                else None
            )
            report = classification_report(
                final_epoch_data["test_labels"],
                final_epoch_data["test_preds"],
                labels=report_labels,
                target_names=report_target_names,
                zero_division=0,
                digits=3,
            )
            ax.text(
                0.01,
                0.99,
                report,
                family="monospace",
                va="top",
                ha="left",
                fontsize=7,
            )
            ax.set_title("Classification Report (Тест, последняя эпоха)", fontsize=10)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Не удалось создать отчет:\n{e}",
                ha="center",
                va="center",
                color="red",
            )

        # Сохранение фигуры
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.suptitle(
            f"Результаты обучения LSTM (Эпох: {cfg.training.training.epochs}, SeqLen: {cfg.training.data.sequence_length}, Hidden: {cfg.training.model.hidden_size})",
            fontsize=16,
        )
        save_path = plot_dir / "lstm_training_metrics.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Графики сохранены в {save_path}")
    except Exception as e:
        warn(f"Не удалось построить или сохранить графики: {e}")
        traceback.print_exc()


# === Функция оценки ===
def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int]]:
    """
    Оценивает производительность обученной модели на тестовом наборе данных.
    """
    print("\n--- Финальная оценка модели на тестовых данных ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Итоговая точность на тесте: {acc:.4f}")
    return all_labels, all_preds


@hydra.main(config_path="../../../../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Главная функция скрипта.

    Args:
        cfg (DictConfig): Конфигурация Hydra со всеми параметрами.
    """
    print("=== Конфигурация LSTM обучения ===")
    print(OmegaConf.to_yaml(cfg.training))
    print("=" * 50)

    print("--- Запуск основного скрипта обучения LSTM ---")

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создание путей
    models_dir = Path(cfg.data.paths.models_dir)
    weights_dir = models_dir / "LSTM" / "LSTM_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = weights_dir / cfg.training.saving.scaler_filename
    best_weights_path = weights_dir / cfg.training.saving.model_filename

    # Вывод параметров запуска
    print(f"--- Параметры Запуска ---")
    print(f"Используемое устройство: {device}")
    print(f"Количество классов: {NUM_CLASSES}")
    print(f"Длина последовательности: {cfg.training.data.sequence_length}")
    print(f"Шаг создания последовательностей: {cfg.training.data.stride}")
    print(f"Признаков на кадр: {cfg.training.data.input_size_per_frame}")
    print(f"Размер скрытого слоя LSTM: {cfg.training.model.hidden_size}")
    print(f"Количество слоев LSTM: {cfg.training.model.num_layers}")
    print(f"Bidirectional LSTM: {cfg.training.model.use_bidirectional}")
    print(f"Размер батча: {cfg.training.training.batch_size}")
    print(f"Количество эпох: {cfg.training.training.epochs}")
    print(f"Скорость обучения: {cfg.training.training.learning_rate}")
    print(f"Weight Decay: {cfg.training.training.weight_decay}")
    print(f"LSTM Dropout: {cfg.training.model.lstm_dropout}")
    print(f"FC Dropout: {cfg.training.model.fc_dropout}")
    print(f"Соотношение Train/Test: {cfg.training.data.train_ratio*100:.1f}%")
    print("-" * 25)

    # Установка seed для воспроизводимости
    set_seed(cfg.training.reproducibility.random_seed)

    # 1. Создание последовательностей данных
    try:
        train_seqs, train_lbls, test_seqs, test_lbls = create_sequences_from_files(
            feature_dir=TRAIN.FEATURES,
            names_structure=NAMES,
            class_map=CLASS_NAME_TO_LABEL_MAP,
            stride=cfg.training.data.stride,
            seq_length=cfg.training.data.sequence_length,
            train_ratio=cfg.training.data.train_ratio,
            input_size_per_frame=cfg.training.data.input_size_per_frame,
        )
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при создании последовательностей:")
        traceback.print_exc()
        return

    # 2. Предобработка данных (Масштабирование)
    print("\n--- Предобработка данных (Масштабирование) ---")
    try:
        num_train_sequences = len(train_seqs)
        if num_train_sequences == 0:
            raise ValueError("Нет обучающих последовательностей для обучения scaler.")

        # Объединяем все кадры из всех обучающих последовательностей
        all_train_frames = torch.cat(train_seqs, dim=0).numpy()
        print(
            f"Форма объединенных обучающих кадров для scaler: {all_train_frames.shape}"
        )

        # Создаем и обучаем Scaler только на обучающих данных
        scaler = StandardScaler()
        scaler.fit(all_train_frames)

        # Сохраняем обученный Scaler
        if cfg.training.saving.save_scaler:
            dump(scaler, scaler_path)
            print(f"Scaler обучен и сохранен в {scaler_path}")

        # Применяем scaler к каждой последовательности
        scaled_train_seqs = [
            torch.tensor(scaler.transform(seq.numpy()), dtype=torch.float32)
            for seq in train_seqs
        ]
        scaled_test_seqs = [
            torch.tensor(scaler.transform(seq.numpy()), dtype=torch.float32)
            for seq in test_seqs
        ]
        print("Масштабирование train и test последовательностей завершено.")
        del all_train_frames, train_seqs, test_seqs

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при масштабировании данных:")
        traceback.print_exc()
        return

    # 3. Создание Датасетов и DataLoader'ов
    print("\n--- Создание Датасетов и DataLoader'ов ---")
    try:
        train_dataset = GaitSequenceDataset(scaled_train_seqs, train_lbls)
        test_dataset = GaitSequenceDataset(scaled_test_seqs, test_lbls)
        del scaled_train_seqs, train_lbls, scaled_test_seqs, test_lbls

        # Создаем генератор для DataLoader
        g = torch.Generator()
        g.manual_seed(cfg.training.reproducibility.random_seed)

        # Создаем DataLoader'ы с параметрами из конфига
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.training.batch_size,
            shuffle=cfg.training.dataloader.shuffle_train,
            num_workers=cfg.training.dataloader.num_workers,
            pin_memory=cfg.training.dataloader.pin_memory and device.type == "cuda",
            generator=g,
            worker_init_fn=seed_worker,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.training.batch_size,
            shuffle=cfg.training.dataloader.shuffle_test,
            num_workers=cfg.training.dataloader.num_workers,
            pin_memory=cfg.training.dataloader.pin_memory and device.type == "cuda",
        )
        print(f"DataLoader'ы созданы. Размер батча: {cfg.training.training.batch_size}")
        print(f"Количество батчей (train): {len(train_loader)}")
        print(f"Количество батчей (test): {len(test_loader)}")

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при создании Dataset/DataLoader:")
        traceback.print_exc()
        return

    # 4. Инициализация модели, функции потерь и оптимизатора
    print("\n--- Инициализация модели, потерь и оптимизатора ---")
    try:
        # Создаем модель LSTM с параметрами из конфига
        model = GaitClassifierLSTM(
            input_size=cfg.training.data.input_size_per_frame,
            hidden_size=cfg.training.model.hidden_size,
            num_layers=cfg.training.model.num_layers,
            num_classes=NUM_CLASSES,
            use_bidirectional=cfg.training.model.use_bidirectional,
            lstm_dropout=cfg.training.model.lstm_dropout,
            use_ffn_head=cfg.training.model.use_ffn_head,
            ffn_hidden_size=cfg.training.model.ffn_hidden_size,
            ffn_dropout=cfg.training.model.fc_dropout,
        ).to(device)

        print("Структура модели:")
        print(model)

        # Выбор функции потерь на основе конфига
        if cfg.training.training.loss.name == "focal":
            criterion = FocalLoss(
                alpha=cfg.training.training.loss.focal.alpha,
                gamma=cfg.training.training.loss.focal.gamma,
            ).to(device)
        else:  # cross_entropy по умолчанию
            criterion = nn.CrossEntropyLoss().to(device)

        print(f"Функция потерь: {criterion.__class__.__name__}")

        # Выбор оптимизатора
        if cfg.training.training.optimizer.name.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.training.training.learning_rate,
                weight_decay=cfg.training.training.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.training.training.learning_rate,
                weight_decay=cfg.training.training.weight_decay,
            )

        print(f"Оптимизатор: {optimizer.__class__.__name__}")

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА при инициализации модели:")
        traceback.print_exc()
        return

    # 5. Обучение модели
    try:
        training_history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            best_model_path=best_weights_path,
            class_names_ordered=CLASS_NAMES_ORDERED,
        )
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА во время обучения модели:")
        traceback.print_exc()
        return

    # 6. Финальная оценка лучшей модели
    print("\n--- Загрузка и финальная оценка лучшей модели ---")
    try:
        # Создаем новую модель и загружаем лучшие веса
        best_model = GaitClassifierLSTM(
            input_size=cfg.training.data.input_size_per_frame,
            hidden_size=cfg.training.model.hidden_size,
            num_layers=cfg.training.model.num_layers,
            num_classes=NUM_CLASSES,
            use_bidirectional=cfg.training.model.use_bidirectional,
            lstm_dropout=cfg.training.model.lstm_dropout,
            use_ffn_head=cfg.training.model.use_ffn_head,
            ffn_hidden_size=cfg.training.model.ffn_hidden_size,
            ffn_dropout=cfg.training.model.fc_dropout,
        ).to(device)

        best_model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"Веса лучшей модели загружены из {best_weights_path}")

        # Финальная оценка
        final_labels, final_preds = evaluate_model(best_model, test_loader, device)

        # Вывод финального отчета
        print("\n--- Финальный отчет по классификации ---")
        report_target_names = (
            [name[:25] for name in CLASS_NAMES_ORDERED] if CLASS_NAMES_ORDERED else None
        )
        report = classification_report(
            final_labels,
            final_preds,
            labels=np.arange(NUM_CLASSES),
            target_names=report_target_names,
            zero_division=0,
            digits=3,
        )
        print(report)

    except FileNotFoundError:
        print(f"Ошибка: Файл с весами не найден: {best_weights_path}")
    except Exception as e:
        print(f"\nОшибка при финальной оценке:")
        traceback.print_exc()

    print("\n--- Скрипт успешно завершен ---")


if __name__ == "__main__":
    main()
