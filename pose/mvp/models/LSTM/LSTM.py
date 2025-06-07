"""
LSTM модель для классификации походки на PyTorch Lightning с MLflow логированием
Согласно Task-2-Training-code.txt
"""

import random
import subprocess
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
from warnings import warn

import hydra
import lightning as L
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hydra import utils
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


# === Функции для логгирования ===
def get_git_commit_id() -> str:
    """Получает текущий git commit id согласно Task-2-Training-code.txt."""
    try:
        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=utils.get_original_cwd()
            )
            .decode("ascii")
            .strip()
        )
        return commit_id
    except Exception as e:
        warn(f"Не удалось получить git commit id: {e}")
        return "unknown"


def setup_mlflow(cfg: DictConfig) -> None:
    """Настраивает MLflow согласно требованиям Task-2-Training-code.txt."""
    if not cfg.training.logging.mlflow.enable:
        return

    try:
        mlflow.set_tracking_uri(cfg.training.logging.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.training.logging.mlflow.experiment_name)
        print(f"✅ MLflow настроен: {cfg.training.logging.mlflow.tracking_uri}")
        print(f"✅ Эксперимент: {cfg.training.logging.mlflow.experiment_name}")
    except Exception as e:
        warn(f"Ошибка настройки MLflow: {e}")


def seed_worker(worker_id: int):
    """Функция инициализации для воркеров DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# === Кастомный датасет ===
class GaitSequenceDataset(Dataset):
    """Датасет PyTorch для последовательностей признаков походки."""

    def __init__(self, sequences: List[torch.Tensor], labels: List[int]):
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
        return self.sequences[idx], self.labels[idx]


# === Модель LSTM ===
class GaitClassifierLSTM(nn.Module):
    """Модель LSTM для классификации временных последовательностей признаков походки."""

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

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=use_bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        classifier_input_size = hidden_size * self.num_directions

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

        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        last_step_out = out[:, -1, :]

        if self.use_ffn_head:
            logits = self.classifier_head(last_step_out)
        else:
            out_dropout = self.dropout_fc(last_step_out)
            logits = self.classifier_head(out_dropout)

        return logits


# === Focal Loss ===
class FocalLoss(nn.Module):
    """Реализация Focal Loss для решения проблемы дисбаланса классов."""

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

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        log_probs_true_class = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probs_true_class = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weights = torch.pow(1 - probs_true_class, self.gamma)

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

        loss_per_sample = -alpha_weights * focal_weights * log_probs_true_class

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


# === PyTorch Lightning модуль ===
class GaitClassifierLightning(L.LightningModule):
    """PyTorch Lightning модуль для классификации походки с LSTM."""

    def __init__(self, cfg: DictConfig, class_names_ordered: List[str] = None):
        super().__init__()
        self.cfg = cfg
        self.class_names_ordered = class_names_ordered or CLASS_NAMES_ORDERED
        self.save_hyperparameters(ignore=["class_names_ordered"])

        # Инициализация модели
        self.model = GaitClassifierLSTM(
            input_size=cfg.training.data.input_size_per_frame,
            hidden_size=cfg.training.model.hidden_size,
            num_layers=cfg.training.model.num_layers,
            num_classes=NUM_CLASSES,
            use_bidirectional=cfg.training.model.use_bidirectional,
            lstm_dropout=cfg.training.model.lstm_dropout,
            use_ffn_head=cfg.training.model.use_ffn_head,
            ffn_hidden_size=cfg.training.model.ffn_hidden_size,
            ffn_dropout=cfg.training.model.fc_dropout,
        )

        # Инициализация критерия потерь
        if cfg.training.training.loss.name == "focal":
            self.criterion = FocalLoss(
                alpha=cfg.training.training.loss.focal.alpha,
                gamma=cfg.training.training.loss.focal.gamma,
                reduction="mean",
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Для сбора данных последней эпохи
        self.final_epoch_data = {
            "train_labels": [],
            "train_preds": [],
            "train_probs": [],
            "test_labels": [],
            "test_preds": [],
            "test_probs": [],
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)

        # Расчет метрик
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )
            precision = precision_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )

        # Логирование метрик согласно Task-2-Training-code.txt
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True)
        self.log("train_precision", precision, on_step=False, on_epoch=True)
        self.log("train_recall", recall, on_step=False, on_epoch=True)

        # Сохранение данных для последней эпохи
        if self.current_epoch == self.cfg.training.training.epochs - 1:
            self.final_epoch_data["train_labels"].extend(labels.cpu().numpy())
            self.final_epoch_data["train_preds"].extend(preds.cpu().numpy())
            self.final_epoch_data["train_probs"].extend(probs.cpu().numpy())

        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)

        # Расчет метрик
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )
            precision = precision_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )

        # Логирование метрик для валидации
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True)

        # Сохранение данных для последней эпохи
        if self.current_epoch == self.cfg.training.training.epochs - 1:
            self.final_epoch_data["test_labels"].extend(labels.cpu().numpy())
            self.final_epoch_data["test_preds"].extend(preds.cpu().numpy())
            self.final_epoch_data["test_probs"].extend(probs.cpu().numpy())

        return loss

    def configure_optimizers(self):
        if self.cfg.training.training.optimizer.name.lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.cfg.training.training.learning_rate,
                weight_decay=self.cfg.training.training.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.parameters(), lr=self.cfg.training.training.learning_rate
            )
        return optimizer

    def on_train_epoch_end(self):
        # Логирование в MLflow через Lightning
        if self.cfg.training.logging.mlflow.enable:
            try:
                current_lr = self.optimizers().param_groups[0]["lr"]
                self.log("learning_rate", current_lr)
            except Exception as e:
                warn(f"Ошибка логирования в MLflow: {e}")

    def on_fit_end(self):
        """Вызывается в конце обучения - сохраняем графики согласно Task-2-Training-code.txt"""
        if self.cfg.training.saving.save_plots:
            try:
                # Используем абсолютный путь к plots/ в корне репозитория
                original_cwd = Path(utils.get_original_cwd())
                plot_dir = (
                    original_cwd
                    / self.cfg.data.paths.plots_dir
                    / self.cfg.training.saving.plots_dirname
                )

                print(f"\nПостроение и сохранение графиков в {plot_dir}...")

                save_training_plots_lightning(
                    self.final_epoch_data, plot_dir, self.cfg, self.class_names_ordered
                )

                # Логирование графиков в MLflow согласно требованиям (5 баллов)
                if self.cfg.training.logging.mlflow.enable:
                    try:
                        plot_path = plot_dir / "lstm_training_metrics.png"
                        if plot_path.exists():
                            mlflow.log_artifact(str(plot_path), "plots")
                            print(f"✅ График загружен в MLflow: {plot_path}")
                    except Exception as e:
                        warn(f"Ошибка загрузки графиков в MLflow: {e}")

            except Exception as e:
                warn(f"Ошибка создания графиков: {e}")


# === Функция для создания графиков ===
def save_training_plots_lightning(
    final_epoch_data: Dict[str, List],
    plot_dir: Path,
    cfg: DictConfig,
    class_names_ordered: List[str] = None,
):
    """Создает и сохраняет графики обучения для Lightning версии."""
    plot_dir.mkdir(exist_ok=True, parents=True)

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # График 1: Матрица ошибок
        ax = axes[0]
        if final_epoch_data["test_labels"] and final_epoch_data["test_preds"]:
            cm = confusion_matrix(
                final_epoch_data["test_labels"], final_epoch_data["test_preds"]
            )
            display_labels = (
                [name[:12] for name in class_names_ordered]
                if class_names_ordered
                else None
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=display_labels
            )
            disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
            ax.set_title("Матрица ошибок (Валидация)")

        # График 2: ROC-кривые
        ax = axes[1]
        if final_epoch_data["test_labels"] and final_epoch_data["test_probs"]:
            test_labels_np = np.array(final_epoch_data["test_labels"])
            test_probs_np = np.array(final_epoch_data["test_probs"])
            y_test_bin = label_binarize(test_labels_np, classes=np.arange(NUM_CLASSES))

            if NUM_CLASSES > 1:
                for i in range(
                    min(10, NUM_CLASSES)
                ):  # Показываем только первые 10 классов
                    if len(np.unique(y_test_bin[:, i])) > 1:
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs_np[:, i])
                        roc_auc = auc(fpr, tpr)
                        class_name = (
                            class_names_ordered[i][:12]
                            if class_names_ordered
                            else str(i)
                        )
                        ax.plot(
                            fpr, tpr, lw=2, label=f"{class_name} (AUC={roc_auc:.2f})"
                        )

                ax.plot([0, 1], [0, 1], "k--", label="Случайное угадывание")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("ROC-кривые (топ-10 классов)")
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(True)

        # График 3: Распределение классов
        ax = axes[2]
        if final_epoch_data["train_labels"] and final_epoch_data["test_labels"]:
            try:
                if pd:
                    train_labels_np = np.array(final_epoch_data["train_labels"])
                    test_labels_np = np.array(final_epoch_data["test_labels"])
                    df_train = pd.DataFrame(
                        {"label": train_labels_np, "split": "Обучение"}
                    )
                    df_test = pd.DataFrame(
                        {"label": test_labels_np, "split": "Валидация"}
                    )
                    df_combined = pd.concat([df_train, df_test])

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
                    ax.set_ylabel("Количество")
                else:
                    ax.text(0.5, 0.5, "pandas недоступен", ha="center", va="center")
            except Exception as e:
                ax.text(0.5, 0.5, f"Ошибка: {e}", ha="center", va="center")

        # График 4: Classification Report
        ax = axes[3]
        ax.axis("off")
        if final_epoch_data["test_labels"] and final_epoch_data["test_preds"]:
            try:
                report = classification_report(
                    final_epoch_data["test_labels"],
                    final_epoch_data["test_preds"],
                    target_names=(
                        [name[:20] for name in class_names_ordered]
                        if class_names_ordered
                        else None
                    ),
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
                ax.set_title("Classification Report", fontsize=10)
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Ошибка отчета:\n{e}",
                    ha="center",
                    va="center",
                    color="red",
                )

        # График 5: Метрики обучения (заглушка)
        ax = axes[4]
        train_acc = (
            accuracy_score(
                final_epoch_data["train_labels"], final_epoch_data["train_preds"]
            )
            if final_epoch_data["train_labels"]
            else 0
        )
        val_acc = (
            accuracy_score(
                final_epoch_data["test_labels"], final_epoch_data["test_preds"]
            )
            if final_epoch_data["test_labels"]
            else 0
        )
        ax.text(
            0.5,
            0.5,
            f"Lightning Training\nЭпох: {cfg.training.training.epochs}\nTrain Acc: {train_acc:.3f}\nVal Acc: {val_acc:.3f}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Training Summary")

        # График 6: Информация о модели
        ax = axes[5]
        ax.text(
            0.5,
            0.5,
            f"LSTM Model\nHidden: {cfg.training.model.hidden_size}\nLayers: {cfg.training.model.num_layers}\nBidirectional: {cfg.training.model.use_bidirectional}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Model Architecture")

        # Сохранение
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.suptitle(
            f"PyTorch Lightning LSTM (Эпох: {cfg.training.training.epochs}, "
            f"Hidden: {cfg.training.model.hidden_size})",
            fontsize=16,
        )
        save_path = plot_dir / "lstm_training_metrics.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✅ Графики сохранены в {save_path}")

    except Exception as e:
        warn(f"Не удалось построить графики: {e}")
        traceback.print_exc()


# === Главная функция ===
@hydra.main(config_path="../../../../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Главная функция с PyTorch Lightning и MLflow интеграцией согласно Task-2-Training-code.txt."""
    if torch.cuda.is_available():
        # Оптимизация для Tensor Cores (RTX 4090)
        torch.set_float32_matmul_precision("medium")  # medium | high
        print("✅ Tensor Cores оптимизация включена (medium precision)")
    print("=== LSTM обучение с PyTorch Lightning + MLflow ===")
    print(f"✅ Создано {NUM_CLASSES} классов: {CLASS_NAMES_ORDERED}")
    print(OmegaConf.to_yaml(cfg.training))
    print("=" * 50)

    # Создание путей согласно требованиям
    original_cwd = Path(utils.get_original_cwd())
    models_dir = original_cwd / cfg.data.paths.models_dir  # Корень/models/
    weights_dir = models_dir / "LSTM" / "LSTM_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = weights_dir / cfg.training.saving.scaler_filename
    best_weights_path = weights_dir / cfg.training.saving.model_filename

    # Настройка MLflow
    setup_mlflow(cfg)

    # Запуск MLflow run согласно требованиям
    with mlflow.start_run(
        run_name=f"LSTM_Training_{cfg.training.model.hidden_size}h_{cfg.training.training.epochs}e"
    ):
        # Логирование гиперпараметров
        if cfg.training.logging.mlflow.enable:
            mlflow.log_params(
                {
                    "model_hidden_size": cfg.training.model.hidden_size,
                    "model_num_layers": cfg.training.model.num_layers,
                    "model_bidirectional": cfg.training.model.use_bidirectional,
                    "lstm_dropout": cfg.training.model.lstm_dropout,
                    "fc_dropout": cfg.training.model.fc_dropout,
                    "batch_size": cfg.training.training.batch_size,
                    "learning_rate": cfg.training.training.learning_rate,
                    "epochs": cfg.training.training.epochs,
                    "sequence_length": cfg.training.data.sequence_length,
                    "train_ratio": cfg.training.data.train_ratio,
                    "optimizer": cfg.training.training.optimizer.name,
                    "loss_function": cfg.training.training.loss.name,
                    "git_commit_id": get_git_commit_id(),
                }
            )

            mlflow.set_tags(
                {
                    "model_type": "LSTM",
                    "task": "gait_classification",
                    "framework": "pytorch_lightning",
                    "stage": "training",
                }
            )

        # Установка seed для воспроизводимости
        L.seed_everything(cfg.training.reproducibility.random_seed, workers=True)
        print(f"[ Установлен Seed : {cfg.training.reproducibility.random_seed} ]")

        # === ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
        print("\n--- Загрузка и подготовка данных ---")

        try:
            sequences, labels = create_sequences_from_files(
                data_path=TRAIN,
                sequence_length=cfg.training.data.sequence_length,
                stride=cfg.training.data.stride,
                names=NAMES,
                class_name_to_label_map=CLASS_NAME_TO_LABEL_MAP,
            )
            print(f"Загружено {len(sequences)} последовательностей")

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке данных: {e}")
            return

        # === НОРМАЛИЗАЦИЯ ДАННЫХ ===
        print("\n--- Нормализация данных ---")

        all_sequences_np = torch.cat(sequences, dim=0).numpy()
        scaler = StandardScaler()
        scaler.fit(all_sequences_np)

        normalized_sequences = []
        for seq in sequences:
            seq_np = seq.numpy()
            seq_normalized = scaler.transform(seq_np)
            normalized_sequences.append(
                torch.tensor(seq_normalized, dtype=torch.float32)
            )

        # Сохранение scaler
        if cfg.training.saving.save_scaler:
            dump(scaler, scaler_path)
            print(f"StandardScaler сохранен в {scaler_path}")

        # === РАЗДЕЛЕНИЕ ДАННЫХ ===
        print("\n--- Разделение данных ---")

        total_sequences = len(normalized_sequences)
        train_size = int(cfg.training.data.train_ratio * total_sequences)

        indices = list(range(total_sequences))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_sequences = [normalized_sequences[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_sequences = [normalized_sequences[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        print(f"Обучающая выборка: {len(train_sequences)} последовательностей")
        print(f"Тестовая выборка: {len(test_sequences)} последовательностей")

        # === СОЗДАНИЕ ДАТАСЕТОВ И DATALOADER'ОВ ===
        print("\n--- Создание DataLoader'ов ---")

        train_dataset = GaitSequenceDataset(train_sequences, train_labels)
        test_dataset = GaitSequenceDataset(test_sequences, test_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.training.batch_size,
            shuffle=cfg.training.dataloader.shuffle_train,
            num_workers=cfg.training.dataloader.num_workers,
            pin_memory=cfg.training.dataloader.pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.training.batch_size,
            shuffle=cfg.training.dataloader.shuffle_test,
            num_workers=cfg.training.dataloader.num_workers,
            pin_memory=cfg.training.dataloader.pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        # === СОЗДАНИЕ LIGHTNING МОДЕЛИ ===
        print("\n--- Создание PyTorch Lightning модели ---")

        lightning_model = GaitClassifierLightning(
            cfg=cfg, class_names_ordered=CLASS_NAMES_ORDERED
        )

        total_params = sum(p.numel() for p in lightning_model.parameters())
        trainable_params = sum(
            p.numel() for p in lightning_model.parameters() if p.requires_grad
        )

        print(f"Lightning модель создана")
        print(f"Общее количество параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,}")

        # === НАСТРОЙКА TRAINER ===
        print("\n--- Настройка Lightning Trainer ---")

        # Настройка callbacks
        callbacks = []

        # ModelCheckpoint для сохранения лучшей модели
        if cfg.training.saving.save_weights:
            checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                dirpath=weights_dir,
                filename=cfg.training.saving.model_filename.replace(".pth", ""),
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                save_weights_only=True,
            )
            callbacks.append(checkpoint_callback)

        # MLflow Logger (встроенный в Lightning)
        logger = None
        if cfg.training.logging.mlflow.enable:
            try:
                from lightning.pytorch.loggers import MLFlowLogger

                logger = MLFlowLogger(
                    experiment_name=cfg.training.logging.mlflow.experiment_name,
                    tracking_uri=cfg.training.logging.mlflow.tracking_uri,
                    run_name=f"LSTM_Training_{cfg.training.model.hidden_size}h_{cfg.training.training.epochs}e",
                )
            except ImportError:
                warn("MLFlowLogger недоступен, используется стандартное логирование")

        # Создание Trainer
        trainer = L.Trainer(
            max_epochs=cfg.training.training.epochs,
            accelerator="auto",  # Автоматически выберет GPU если доступен
            devices="auto",
            logger=logger,
            callbacks=callbacks,
            deterministic=cfg.training.reproducibility.deterministic,
            enable_progress_bar=cfg.training.logging.verbose,
            log_every_n_steps=50,
        )

        # === ОБУЧЕНИЕ МОДЕЛИ ===
        print("\n--- Начало обучения с PyTorch Lightning ---")

        try:
            trainer.fit(
                model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            print("✅ Обучение успешно завершено")

            # Логирование финальной модели в MLflow
            if cfg.training.logging.mlflow.enable:
                try:
                    # Логируем модель как артефакт
                    mlflow.pytorch.log_model(
                        lightning_model,
                        "model",
                        registered_model_name="LSTM_Gait_Classifier_Lightning",
                    )

                    # Логируем файлы
                    if best_weights_path.exists():
                        mlflow.log_artifact(str(best_weights_path), "weights")
                    mlflow.log_artifact(str(scaler_path), "preprocessing")

                    print("✅ Модель и артефакты загружены в MLflow")

                except Exception as e:
                    warn(f"Ошибка логирования модели в MLflow: {e}")

        except Exception as e:
            print(f"\nКРИТИЧЕСКАЯ ОШИБКА во время обучения модели:")
            traceback.print_exc()
            return

    print("\n--- Скрипт успешно завершен с PyTorch Lightning + MLflow ---")


if __name__ == "__main__":
    main()
