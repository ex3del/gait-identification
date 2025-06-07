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


# === Функция сохранения отдельных графиков ===
def save_individual_training_plots(
    lightning_model: "GaitClassifierLightning",
    plot_dir: Path,
    cfg: DictConfig,
    class_names_ordered: List[str] = None,
):
    """Создает отдельные графики для каждой метрики согласно Task-2-Training-code.txt."""
    plot_dir.mkdir(exist_ok=True, parents=True)
    print(f"📊 Сохранение графиков в {plot_dir}...")

    try:
        # Получаем данные из модели
        history = getattr(lightning_model, "training_history", {})
        final_epoch_data = getattr(lightning_model, "final_epoch_data", {})

        plot_paths = []

        # === График 1: Loss (отдельный файл) ===
        plt.figure(figsize=(10, 6))

        if history.get("train_loss") and history.get("test_loss"):
            # Синхронизируем размеры массивов
            train_loss = history["train_loss"]
            test_loss = history["test_loss"]
            min_len = min(len(train_loss), len(test_loss))

            if min_len > 0:
                epochs = range(1, min_len + 1)
                plt.plot(
                    epochs,
                    train_loss[:min_len],
                    "b-",
                    label="Training Loss",
                    linewidth=2,
                )
                plt.plot(
                    epochs,
                    test_loss[:min_len],
                    "r-",
                    label="Validation Loss",
                    linewidth=2,
                )
                plt.xlabel("Эпоха")
                plt.ylabel("Потери")
                plt.title("История потерь обучения")
                plt.legend()
                plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "История потерь недоступна",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.title("История потерь обучения")

        loss_path = plot_dir / "loss_history.png"
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(loss_path)
        print(f"✅ График потерь: {loss_path}")

        # === График 2: Accuracy (отдельный файл) ===
        plt.figure(figsize=(10, 6))

        if history.get("train_acc") and history.get("test_acc"):
            # Синхронизируем размеры массивов
            train_acc = history["train_acc"]
            test_acc = history["test_acc"]
            min_len = min(len(train_acc), len(test_acc))

            if min_len > 0:
                epochs = range(1, min_len + 1)
                plt.plot(
                    epochs,
                    train_acc[:min_len],
                    "g-",
                    label="Training Accuracy",
                    linewidth=2,
                )
                plt.plot(
                    epochs,
                    test_acc[:min_len],
                    "m-",
                    label="Validation Accuracy",
                    linewidth=2,
                )
                plt.xlabel("Эпоха")
                plt.ylabel("Точность")
                plt.title("История точности обучения")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim([0, 1])
        else:
            plt.text(
                0.5,
                0.5,
                "История точности недоступна",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.title("История точности обучения")

        accuracy_path = plot_dir / "accuracy_history.png"
        plt.savefig(accuracy_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(accuracy_path)
        print(f"✅ График точности: {accuracy_path}")

        # === График 3: F1 Score (отдельный файл) ===
        plt.figure(figsize=(10, 6))

        if history.get("train_f1") and history.get("test_f1"):
            # Синхронизируем размеры массивов
            train_f1 = history["train_f1"]
            test_f1 = history["test_f1"]
            min_len = min(len(train_f1), len(test_f1))

            if min_len > 0:
                epochs = range(1, min_len + 1)
                plt.plot(
                    epochs,
                    train_f1[:min_len],
                    "orange",
                    label="Training F1",
                    linewidth=2,
                )
                plt.plot(
                    epochs,
                    test_f1[:min_len],
                    "purple",
                    label="Validation F1",
                    linewidth=2,
                )
                plt.xlabel("Эпоха")
                plt.ylabel("F1 Score")
                plt.title("История F1 Score")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim([0, 1])
        else:
            plt.text(
                0.5, 0.5, "История F1 недоступна", ha="center", va="center", fontsize=14
            )
            plt.title("История F1 Score")

        f1_path = plot_dir / "f1_history.png"
        plt.savefig(f1_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(f1_path)
        print(f"✅ График F1: {f1_path}")

        # === График 4: Confusion Matrix (отдельный файл) ===
        plt.figure(figsize=(12, 10))

        if final_epoch_data.get("test_labels") and final_epoch_data.get("test_preds"):
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
            disp.plot(cmap="Blues", xticks_rotation="vertical")
            plt.title("Матрица ошибок (Валидация)")
        else:
            plt.text(
                0.5,
                0.5,
                "Данные для матрицы ошибок недоступны",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.title("Матрица ошибок (Валидация)")

        confusion_path = plot_dir / "confusion_matrix.png"
        plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(confusion_path)
        print(f"✅ Матрица ошибок: {confusion_path}")

        # === График 5: Classification Report (отдельный файл) ===
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

        if final_epoch_data.get("test_labels") and final_epoch_data.get("test_preds"):
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
                    fontsize=10,
                )
                ax.set_title("Classification Report", fontsize=14, pad=20)
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Ошибка создания отчета:\n{e}",
                    ha="center",
                    va="center",
                    color="red",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Данные для отчета недоступны",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_title("Classification Report", fontsize=14)

        report_path = plot_dir / "classification_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(report_path)
        print(f"✅ Classification Report: {report_path}")

        # === График 6: Model Summary (отдельный файл) ===
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        total_params = sum(p.numel() for p in lightning_model.parameters())
        trainable_params = sum(
            p.numel() for p in lightning_model.parameters() if p.requires_grad
        )

        summary_text = f"""
PyTorch Lightning LSTM Model Summary

Architecture:
├─ Hidden Size: {cfg.training.model.hidden_size}
├─ Num Layers: {cfg.training.model.num_layers}
├─ Bidirectional: {cfg.training.model.use_bidirectional}
├─ LSTM Dropout: {cfg.training.model.lstm_dropout}
├─ FC Dropout: {cfg.training.model.fc_dropout}

Training Config:
├─ Epochs: {cfg.training.training.epochs}
├─ Batch Size: {cfg.training.training.batch_size}
├─ Learning Rate: {cfg.training.training.learning_rate}
├─ Optimizer: {cfg.training.training.optimizer.name}

Data Config:
├─ Sequence Length: {cfg.training.data.sequence_length}
├─ Input Size per Frame: {cfg.training.data.input_size_per_frame}
├─ Stride: {cfg.training.data.stride}
├─ Train Ratio: {cfg.training.data.train_ratio}

Model Parameters:
├─ Total Parameters: {total_params:,}
├─ Trainable Parameters: {trainable_params:,}
├─ Model Size: {total_params * 4 / 1024 / 1024:.2f} MB

Classes: {NUM_CLASSES} gait classes
        """

        ax.text(
            0.05,
            0.95,
            summary_text,
            family="monospace",
            va="top",
            ha="left",
            fontsize=10,
        )
        ax.set_title("Model & Training Summary", fontsize=14, pad=20)

        summary_path = plot_dir / "model_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(summary_path)
        print(f"✅ Model Summary: {summary_path}")

        print(f"📊 Все графики сохранены в {plot_dir}")
        return plot_paths

    except Exception as e:
        warn(f"Ошибка создания графиков: {e}")
        traceback.print_exc()
        return []


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

        # ✅ ИСПРАВЛЕННЫЙ СБОР ИСТОРИИ МЕТРИК
        self.training_history = {
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

        # Временные переменные для накопления метрик за эпоху
        self.epoch_train_metrics = {
            "loss": [],
            "acc": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }
        self.epoch_val_metrics = {
            "loss": [],
            "acc": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

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

        # ✅ НАКАПЛИВАЕМ МЕТРИКИ ЗА ЭПОХУ
        self.epoch_train_metrics["loss"].append(loss.item())
        self.epoch_train_metrics["acc"].append(acc)
        self.epoch_train_metrics["f1"].append(f1)
        self.epoch_train_metrics["precision"].append(precision)
        self.epoch_train_metrics["recall"].append(recall)

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

        # ✅ НАКАПЛИВАЕМ МЕТРИКИ ЗА ЭПОХУ
        self.epoch_val_metrics["loss"].append(loss.item())
        self.epoch_val_metrics["acc"].append(acc)
        self.epoch_val_metrics["f1"].append(f1)
        self.epoch_val_metrics["precision"].append(precision)
        self.epoch_val_metrics["recall"].append(recall)

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

    # ✅ ИСПРАВЛЕННЫЙ СБОР МЕТРИК ПО ЭПОХАМ (ПРОПУСКАЕМ SANITY CHECK)
    def on_train_epoch_end(self):
        """Собираем усредненные метрики обучения за эпоху"""
        # ✅ ПРОПУСКАЕМ SANITY CHECK
        if self.trainer.sanity_checking:
            return

        if self.epoch_train_metrics["loss"]:
            # Усредняем метрики за эпоху
            avg_train_loss = np.mean(self.epoch_train_metrics["loss"])
            avg_train_acc = np.mean(self.epoch_train_metrics["acc"])
            avg_train_f1 = np.mean(self.epoch_train_metrics["f1"])
            avg_train_precision = np.mean(self.epoch_train_metrics["precision"])
            avg_train_recall = np.mean(self.epoch_train_metrics["recall"])

            # Добавляем в историю
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["train_acc"].append(avg_train_acc)
            self.training_history["train_f1"].append(avg_train_f1)
            self.training_history["train_precision"].append(avg_train_precision)
            self.training_history["train_recall"].append(avg_train_recall)

            # Очищаем временные метрики
            self.epoch_train_metrics = {
                "loss": [],
                "acc": [],
                "f1": [],
                "precision": [],
                "recall": [],
            }

            print(
                f"Эпоха {self.current_epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}"
            )

        # Логирование в MLflow через Lightning
        if self.cfg.training.logging.mlflow.enable:
            try:
                current_lr = self.optimizers().param_groups[0]["lr"]
                self.log("learning_rate", current_lr)
            except Exception as e:
                warn(f"Ошибка логирования в MLflow: {e}")

    def on_validation_epoch_end(self):
        """Собираем усредненные метрики валидации за эпоху"""
        # ✅ ПРОПУСКАЕМ SANITY CHECK
        if self.trainer.sanity_checking:
            return

        if self.epoch_val_metrics["loss"]:
            # Усредняем метрики за эпоху
            avg_val_loss = np.mean(self.epoch_val_metrics["loss"])
            avg_val_acc = np.mean(self.epoch_val_metrics["acc"])
            avg_val_f1 = np.mean(self.epoch_val_metrics["f1"])
            avg_val_precision = np.mean(self.epoch_val_metrics["precision"])
            avg_val_recall = np.mean(self.epoch_val_metrics["recall"])

            # Добавляем в историю
            self.training_history["test_loss"].append(avg_val_loss)
            self.training_history["test_acc"].append(avg_val_acc)
            self.training_history["test_f1"].append(avg_val_f1)
            self.training_history["test_precision"].append(avg_val_precision)
            self.training_history["test_recall"].append(avg_val_recall)

            # Очищаем временные метрики
            self.epoch_val_metrics = {
                "loss": [],
                "acc": [],
                "f1": [],
                "precision": [],
                "recall": [],
            }

            print(
                f"Эпоха {self.current_epoch}: Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}"
            )

    def on_fit_end(self):
        """Вызывается в конце обучения - сохраняем графики согласно Task-2-Training-code.txt"""
        if self.cfg.training.saving.save_plots:
            try:
                original_cwd = Path(utils.get_original_cwd())
                plot_dir = (
                    original_cwd
                    / self.cfg.data.paths.plots_dir
                    / self.cfg.training.saving.plots_dirname
                )

                print(f"\n📊 Построение и сохранение графиков в {plot_dir}...")

                plot_paths = save_individual_training_plots(
                    self, plot_dir, self.cfg, self.class_names_ordered
                )

                # ✅ ИСПРАВЛЕННОЕ ЛОГИРОВАНИЕ ГРАФИКОВ В MLFLOW
                if self.cfg.training.logging.mlflow.enable and hasattr(
                    self.logger, "experiment"
                ):
                    try:
                        # Получаем текущий run_id из Lightning logger
                        current_run = self.logger.experiment.get_run(self.logger.run_id)

                        # Логируем каждый график отдельно с правильным API
                        for plot_path in plot_paths:
                            if plot_path.exists():
                                # ✅ ПРАВИЛЬНЫЙ ВЫЗОВ - используем run_id из logger
                                self.logger.experiment.log_artifact(
                                    run_id=self.logger.run_id,
                                    local_path=str(plot_path),
                                    artifact_path="plots",
                                )
                                print(f"✅ График загружен в MLflow: {plot_path.name}")
                    except Exception as e:
                        warn(f"Ошибка загрузки графиков в MLflow: {e}")
                        # Альтернативный способ через стандартный mlflow
                        try:
                            for plot_path in plot_paths:
                                if plot_path.exists():
                                    mlflow.log_artifact(str(plot_path), "plots")
                                    print(
                                        f"✅ График загружен в MLflow (fallback): {plot_path.name}"
                                    )
                        except Exception as fallback_error:
                            warn(f"Fallback также не сработал: {fallback_error}")

            except Exception as e:
                warn(f"Ошибка создания графиков: {e}")


# Обновите функцию export_model_to_onnx для использования конфигурации:


def export_model_to_onnx(
    lightning_model: GaitClassifierLightning, cfg: DictConfig, original_cwd: Path
) -> Path:
    """
    Экспортирует обученную Lightning модель в ONNX формат согласно Task-2-Training-code.txt.
    """
    print("🔄 Экспорт LSTM модели в ONNX...")

    # Пути для сохранения
    models_dir = original_cwd / cfg.data.paths.models_dir
    onnx_dir = models_dir / "LSTM" / "ONNX"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = onnx_dir / "lstm_gait_classifier.onnx"

    # Переводим модель в режим оценки
    lightning_model.eval()

    # Создание примера входных данных
    example_input = torch.randn(
        1,  # batch_size = 1 для примера
        cfg.training.data.sequence_length,  # 30
        cfg.training.data.input_size_per_frame,  # 84
    )

    print(f"📝 Форма входных данных: {example_input.shape}")
    print(f"📁 Сохранение в: {onnx_path}")

    # ✅ ИСПОЛЬЗУЕМ ПАРАМЕТРЫ ИЗ КОНФИГУРАЦИИ
    opset_version = (
        getattr(cfg.training, "production", {}).get("onnx", {}).get("opset_version", 11)
    )
    optimize = (
        getattr(cfg.training, "production", {}).get("onnx", {}).get("optimize", True)
    )

    # Экспорт в ONNX
    with torch.no_grad():
        torch.onnx.export(
            lightning_model.model,  # Используем внутреннюю LSTM модель
            example_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,  # Из конфигурации
            do_constant_folding=optimize,  # Из конфигурации
            input_names=["input_sequences"],
            output_names=["class_predictions"],
            dynamic_axes={
                "input_sequences": {0: "batch_size"},  # Динамический batch size
                "class_predictions": {0: "batch_size"},
            },
            verbose=False,
        )

    print(f"✅ ONNX модель сохранена: {onnx_path}")

    # Проверка созданной модели
    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX модель прошла проверку")

        # Информация о модели
        file_size = onnx_path.stat().st_size / 1024 / 1024
        print(f"📊 Размер ONNX файла: {file_size:.2f} MB")
        print(f"📊 ONNX opset version: {opset_version}")

    except ImportError:
        warn("ONNX библиотека не установлена для проверки модели")
    except Exception as e:
        warn(f"Ошибка проверки ONNX модели: {e}")

    return onnx_path


# === Главная функция ===
@hydra.main(config_path="../../../../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Главная функция с PyTorch Lightning и MLflow интеграцией согласно Task-2-Training-code.txt."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"✅ GPU оптимизация включена для {device_name} (medium precision)")

    print("=== LSTM обучение с PyTorch Lightning + MLflow ===")
    print(f"✅ Создано {NUM_CLASSES} классов: {CLASS_NAMES_ORDERED}")
    print(OmegaConf.to_yaml(cfg.training))
    print("=" * 50)

    # Создание путей согласно требованиям
    original_cwd = Path(utils.get_original_cwd())
    models_dir = original_cwd / cfg.data.paths.models_dir
    weights_dir = models_dir / "LSTM" / "LSTM_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = weights_dir / cfg.training.saving.scaler_filename
    best_weights_path = weights_dir / cfg.training.saving.model_filename

    # Настройка MLflow (только настройка URI, без создания run)
    setup_mlflow(cfg)

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
        normalized_sequences.append(torch.tensor(seq_normalized, dtype=torch.float32))

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

    # ✅ ИСПОЛЬЗУЕМ ТОЛЬКО Lightning MLFlowLogger
    logger = None
    if cfg.training.logging.mlflow.enable:
        try:
            from lightning.pytorch.loggers import MLFlowLogger

            logger = MLFlowLogger(
                experiment_name=cfg.training.logging.mlflow.experiment_name,
                tracking_uri=cfg.training.logging.mlflow.tracking_uri,
                run_name=f"LSTM_Training_{cfg.training.model.hidden_size}h_{cfg.training.training.epochs}e",
                # ✅ ДОБАВЛЯЕМ ТЕГИ И ПАРАМЕТРЫ ЧЕРЕЗ LOGGER
                tags={
                    "model_type": "LSTM",
                    "task": "gait_classification",
                    "framework": "pytorch_lightning",
                    "stage": "training",
                },
            )

            # ✅ ЛОГИРУЕМ ГИПЕРПАРАМЕТРЫ ЧЕРЕЗ LOGGER
            logger.log_hyperparams(
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

        except ImportError:
            warn("MLFlowLogger недоступен, используется стандартное логирование")

    # Создание Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.training.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,  # ✅ ПЕРЕДАЕМ LIGHTNING LOGGER
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

        # ✅ ЛОГИРОВАНИЕ АРТЕФАКТОВ ЧЕРЕЗ LIGHTNING LOGGER
        if cfg.training.logging.mlflow.enable and logger:
            try:
                from mlflow.models import infer_signature

                # Создаем пример входных данных
                example_input = torch.randn(
                    1,  # batch_size
                    cfg.training.data.sequence_length,  # 30
                    cfg.training.data.input_size_per_frame,  # 84
                )

                # Получаем пример выходных данных
                lightning_model.eval()
                with torch.no_grad():
                    example_output = lightning_model(example_input)

                signature = infer_signature(
                    example_input.numpy(), example_output.numpy()
                )

                mlflow.pytorch.log_model(
                    lightning_model,
                    "model",
                    registered_model_name="LSTM_Gait_Classifier_Lightning",
                    signature=signature,
                    input_example=example_input.numpy(),
                )

                print("✅ Модель сохранена с signature")

            except Exception as e:
                warn(f"Ошибка логирования модели в MLflow: {e}")

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА во время обучения модели:")
        traceback.print_exc()
        return

    # === ЭКСПОРТ В PRODUCTION ФОРМАТЫ ===
    production_config = getattr(cfg.training, "production", {})
    onnx_config = production_config.get("onnx", {})

    if cfg.training.saving.save_weights and onnx_config.get("enable", True):
        try:
            # Экспорт в ONNX
            onnx_path = export_model_to_onnx(lightning_model, cfg, original_cwd)

            # Логирование ONNX в MLflow
            if cfg.training.logging.mlflow.enable:
                mlflow.log_artifact(str(onnx_path), "models")
                print(f"✅ ONNX модель загружена в MLflow: {onnx_path}")

            print("✅ Production экспорт в ONNX завершен")

        except Exception as e:
            warn(f"Ошибка экспорта в ONNX: {e}")
            traceback.print_exc()

    print("\n--- Скрипт успешно завершен с ONNX экспортом ---")
    print("\n--- Скрипт успешно завершен с PyTorch Lightning + MLflow ---")


if __name__ == "__main__":
    main()
