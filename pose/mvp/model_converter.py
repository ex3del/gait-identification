"""
Python CLI для конвертации ONNX модели в TensorRT
Согласно Task-2-Training-code.txt - Model production packaging (5 баллов)

Использование:
    uv run python -m pose.mvp.model_converter
    uv run python -m pose.mvp.model_converter --precision fp32 --opt-batch 16
    uv run python -m pose.mvp.model_converter --config configs/training/lstm.yaml
"""

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def get_git_commit_id() -> str:
    """Получает текущий git commit id."""
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
            .decode("ascii")
            .strip()
        )
        return commit_id
    except Exception:
        return "unknown"


def check_tensorrt_installation() -> Tuple[bool, str]:
    """Проверяет установку TensorRT и trtexec."""
    try:
        result = subprocess.run(
            ["trtexec", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Извлекаем версию из вывода
            version_line = result.stdout.split("\n")[0] if result.stdout else "Unknown"
            return True, version_line
        else:
            return False, "trtexec недоступен"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "trtexec не найден в PATH"


def validate_precision(precision: str) -> str:
    """Валидирует и возвращает флаги точности для trtexec."""
    precision = precision.lower()
    precision_map = {"fp32": "", "fp16": "--fp16", "int8": "--int8"}

    if precision not in precision_map:
        raise ValueError(
            f"Неподдерживаемая точность: {precision}. Доступны: {list(precision_map.keys())}"
        )

    return precision_map[precision]


def build_shapes(
    min_batch: int, opt_batch: int, max_batch: int, cfg: DictConfig
) -> Tuple[str, str, str]:
    """Строит строки форм тензоров для LSTM модели из конфигурации."""
    sequence_length = cfg.training.data.sequence_length
    input_size = cfg.training.data.input_size_per_frame

    min_shape = f"input_sequences:{min_batch}x{sequence_length}x{input_size}"
    opt_shape = f"input_sequences:{opt_batch}x{sequence_length}x{input_size}"
    max_shape = f"input_sequences:{max_batch}x{sequence_length}x{input_size}"

    return min_shape, opt_shape, max_shape


def load_config(config_path: Optional[Path] = None) -> DictConfig:
    """
    Загружает конфигурацию через Hydra с правильными путями.
    Скрипт лежит в pose/mvp/, configs в корне проекта.
    """
    GlobalHydra.instance().clear()

    if config_path:
        # ✅ ИСПРАВЛЕННАЯ ЗАГРУЗКА КАСТОМНОГО КОНФИГА
        # Если путь абсолютный, используем его напрямую
        if config_path.is_absolute():
            config_dir = str(config_path.parent)
            config_name = config_path.stem
        else:
            # Если относительный, строим от корня проекта
            # pose/mvp/ -> ../../ для выхода в корень проекта
            project_root = Path(__file__).parent.parent.parent  # ml-project/
            full_config_path = project_root / config_path
            config_dir = str(full_config_path.parent)
            config_name = full_config_path.stem

        with initialize(config_path=config_dir, version_base="1.1"):
            cfg = compose(config_name=config_name)
    else:
        # ✅ ИСПРАВЛЕННАЯ ЗАГРУЗКА ДЕФОЛТНОГО КОНФИГА
        # Путь от pose/mvp/ до configs/ - это ../../configs
        config_path_relative = "../../configs"

        with initialize(config_path=config_path_relative, version_base="1.1"):
            cfg = compose(config_name="config")

    return cfg


def create_sample_data(cfg: DictConfig, output_path: Path, num_samples: int = 10):
    """Создает пример данных для тестирования TensorRT engine согласно Task-2-Training-code.txt."""
    print(f"📝 Создание примера данных: {output_path}")

    sequence_length = cfg.training.data.sequence_length
    input_size = cfg.training.data.input_size_per_frame

    # Создание случайных данных в формате LSTM (нормализованных как после StandardScaler)
    sample_data = torch.randn(
        num_samples, sequence_length, input_size, dtype=torch.float32
    )

    # Ограничиваем значения как после StandardScaler
    sample_data = torch.clamp(sample_data, -3.0, 3.0)

    # Создание метаданных
    metadata = {
        "batch_size": num_samples,
        "sequence_length": sequence_length,
        "input_size_per_frame": input_size,
        "description": "Sample data for LSTM gait classification TensorRT testing",
        "created_by": "model_converter.py",
        "git_commit": get_git_commit_id(),
        "data_range": [-3.0, 3.0],
        "data_format": "standardized",
    }

    # Сохранение
    torch.save({"input_sequences": sample_data, "metadata": metadata}, output_path)

    print(f"✅ Сохранено {num_samples} примеров формы {list(sample_data.shape)}")
    return output_path


def convert_onnx_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    cfg: DictConfig,
    precision: Optional[str] = None,
    min_batch: Optional[int] = None,
    opt_batch: Optional[int] = None,
    max_batch: Optional[int] = None,
    workspace_size: Optional[int] = None,
    verbose: bool = False,
    benchmark: bool = True,
) -> bool:
    """
    Конвертирует ONNX модель в TensorRT engine используя конфигурацию.
    """
    try:
        # Проверка входного файла
        if not onnx_path.exists():
            print(f"❌ ONNX файл не найден: {onnx_path}")
            return False

        # Создание выходной директории
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # Получение параметров из конфигурации с возможностью переопределения
        tensorrt_cfg = cfg.training.production.tensorrt

        used_precision = precision or tensorrt_cfg.precision
        used_workspace = workspace_size or tensorrt_cfg.workspace_size
        used_min_batch = min_batch or 1
        used_opt_batch = opt_batch or cfg.training.training.batch_size
        used_max_batch = max_batch or tensorrt_cfg.max_batch_size

        # Валидация точности
        precision_flag = validate_precision(used_precision)

        # Построение форм тензоров
        min_shape, opt_shape, max_shape = build_shapes(
            used_min_batch, used_opt_batch, used_max_batch, cfg
        )

        print("=== Конвертация ONNX в TensorRT ===")
        print(f"📁 ONNX модель: {onnx_path}")
        print(f"📁 TensorRT engine: {engine_path}")
        print(f"⚙️ Точность: {used_precision}")
        print(f"💾 Workspace: {used_workspace}MB")
        print(
            f"📊 Batch размеры: min={used_min_batch}, opt={used_opt_batch}, max={used_max_batch}"
        )
        print(f"📐 Формы тензоров:")
        print(f"   Минимальная: {min_shape}")
        print(f"   Оптимальная: {opt_shape}")
        print(f"   Максимальная: {max_shape}")
        print("=" * 50)

        # Команда конвертации
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={used_workspace}",
            f"--minShapes={min_shape}",
            f"--optShapes={opt_shape}",
            f"--maxShapes={max_shape}",
            "--buildOnly",
        ]

        # Добавление флагов точности
        if precision_flag:
            cmd.append(precision_flag)

        # Добавление verbose
        if verbose:
            cmd.append("--verbose")

        print("🔄 Начало конвертации...")
        print(f"🔧 Команда: {' '.join(cmd)}")

        # Выполнение конвертации
        result = subprocess.run(cmd, capture_output=not verbose, text=True)

        if result.returncode == 0:
            print("✅ Конвертация успешно завершена!")

            # Информация о размере
            if engine_path.exists():
                size_mb = engine_path.stat().st_size / 1024 / 1024
                print(f"📊 Размер engine: {size_mb:.2f} MB")

                # Бенчмарк
                if benchmark:
                    print("\n🚀 Запуск бенчмарка производительности...")
                    benchmark_cmd = [
                        "trtexec",
                        f"--loadEngine={engine_path}",
                        f"--batch={used_opt_batch}",
                        "--iterations=100",
                        "--warmUp=10",
                        "--duration=10",
                    ]

                    benchmark_result = subprocess.run(
                        benchmark_cmd, capture_output=True, text=True
                    )
                    if benchmark_result.returncode == 0:
                        print("✅ Бенчмарк завершен")
                        # Извлечение статистик из вывода
                        for line in benchmark_result.stdout.split("\n"):
                            if "mean" in line.lower() and (
                                "ms" in line.lower() or "throughput" in line.lower()
                            ):
                                print(f"📈 {line.strip()}")
                    else:
                        print("⚠️ Ошибка бенчмарка (не критично)")

                return True
            else:
                print("❌ Engine файл не создан")
                return False
        else:
            print("❌ Ошибка конвертации:")
            if not verbose:
                print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()
        return False


def main():
    """CLI интерфейс для конвертации ONNX в TensorRT."""
    parser = argparse.ArgumentParser(
        description="Конвертация ONNX модели в TensorRT согласно Task-2-Training-code.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовая конвертация (использует настройки из конфига)
  uv run python -m pose.mvp.model_converter

  # Кастомная точность и размеры
  uv run python -m pose.mvp.model_converter --precision fp32 --opt-batch 16

  # Кастомный конфиг
  uv run python -m pose.mvp.model_converter --config configs/training/lstm.yaml

  # Максимальная производительность
  uv run python -m pose.mvp.model_converter --precision fp16 --max-batch 128 --workspace-size 2048

  # Создание примера данных для DVC
  uv run python -m pose.mvp.model_converter --create-sample-data
        """,
    )

    # Основные параметры
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к конфигурации (по умолчанию: configs/config.yaml)",
    )

    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Путь к ONNX файлу (по умолчанию: models/LSTM/ONNX/lstm_gait_classifier.onnx)",
    )

    parser.add_argument(
        "--engine-path",
        type=Path,
        default=None,
        help="Путь для TensorRT engine (по умолчанию: models/LSTM/TensorRT/lstm_gait_classifier.engine)",
    )

    # Переопределения параметров из конфига
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default=None,
        help="Точность модели (переопределяет конфиг)",
    )

    parser.add_argument(
        "--workspace-size",
        type=int,
        default=None,
        help="Размер workspace в MB (переопределяет конфиг)",
    )

    parser.add_argument(
        "--min-batch",
        type=int,
        default=None,
        help="Минимальный batch size (переопределяет конфиг)",
    )

    parser.add_argument(
        "--opt-batch",
        type=int,
        default=None,
        help="Оптимальный batch size (переопределяет конфиг)",
    )

    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Максимальный batch size (переопределяет конфиг)",
    )

    # Опции
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")

    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Не запускать бенчмарк после конвертации",
    )

    # Создание примера данных для DVC
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Создать пример данных для тестирования (добавить в DVC)",
    )

    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/tensorrt_samples/sample_batch_32.pt"),
        help="Путь для сохранения примера данных",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="Количество примеров в sample данных",
    )

    args = parser.parse_args()

    # Загрузка конфигурации
    try:
        print("=== Загрузка конфигурации ===")
        cfg = load_config(args.config)
        print(f"📁 Конфигурация: {args.config or 'configs/config.yaml'}")
        print(f"⚙️ Параметры из конфига:")
        print(f"   Precision: {cfg.training.production.tensorrt.precision}")
        print(f"   Workspace: {cfg.training.production.tensorrt.workspace_size}MB")
        print(f"   Max batch: {cfg.training.production.tensorrt.max_batch_size}")
        print(
            f"   LSTM: {cfg.training.data.sequence_length}x{cfg.training.data.input_size_per_frame}"
        )
        print("=" * 50)
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        print(f"💡 Проверьте структуру проекта:")
        print(f"   Скрипт: {Path(__file__)}")
        print(f"   Configs: {Path(__file__).parent.parent.parent / 'configs'}")
        traceback.print_exc()
        return 1

    # ✅ ИСПРАВЛЕННЫЕ ПУТИ К ФАЙЛАМ
    # Определяем корень проекта от расположения скрипта
    project_root = Path(__file__).parent.parent.parent  # ml-project/

    if args.onnx_path is None:
        onnx_path = project_root / "models/LSTM/ONNX/lstm_gait_classifier.onnx"
    else:
        onnx_path = args.onnx_path
        if not onnx_path.is_absolute():
            onnx_path = project_root / onnx_path

    if args.engine_path is None:
        engine_path = project_root / "models/LSTM/TensorRT/lstm_gait_classifier.engine"
    else:
        engine_path = args.engine_path
        if not engine_path.is_absolute():
            engine_path = project_root / engine_path

    # Проверка TensorRT
    tensorrt_available, tensorrt_info = check_tensorrt_installation()
    if not tensorrt_available:
        print(f"❌ {tensorrt_info}")
        print("\n💡 Для установки TensorRT:")
        print("   1. Скачайте TensorRT с NVIDIA Developer")
        print("   2. Добавьте в PATH: export PATH=$PATH:/path/to/tensorrt/bin")
        print("   3. Установите Python пакет: pip install tensorrt")
        return 1
    else:
        print(f"✅ TensorRT: {tensorrt_info}")

    # Создание примера данных (если запрошено)
    if args.create_sample_data:
        try:
            sample_output = args.sample_output
            if not sample_output.is_absolute():
                sample_output = project_root / sample_output

            sample_output.parent.mkdir(parents=True, exist_ok=True)
            sample_path = create_sample_data(cfg, sample_output, args.sample_size)
            print(f"\n💡 Пример данных создан: {sample_path}")
            print(f"📦 Добавьте в DVC:")
            print(f"   dvc add {sample_path.relative_to(project_root)}")
            print(f"   git add {sample_path.relative_to(project_root)}.dvc")
            print(f"   git commit -m 'Add TensorRT sample data'")
        except Exception as e:
            print(f"❌ Ошибка создания примера данных: {e}")
            return 1

    # Проверка ONNX файла
    if not onnx_path.exists():
        print(f"❌ ONNX файл не найден: {onnx_path}")
        print("💡 Сначала обучите модель и экспортируйте в ONNX:")
        print("   uv run python -m pose.mvp.models.LSTM.LSTM")
        return 1

    # Конвертация
    success = convert_onnx_to_tensorrt(
        onnx_path=onnx_path,
        engine_path=engine_path,
        cfg=cfg,
        precision=args.precision,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_size=args.workspace_size,
        verbose=args.verbose,
        benchmark=not args.no_benchmark,
    )

    if success:
        print("\n🎯 Конвертация завершена успешно!")
        print(f"📁 TensorRT engine: {engine_path}")
        print(f"📊 Размер файла: {engine_path.stat().st_size / 1024 / 1024:.2f} MB")
        print("\n💡 Следующие шаги:")
        print("   1. Добавьте engine в .gitignore (большой файл)")
        print("   2. Используйте engine для inference через TensorRT API")
        print("   3. Настройте production inference server")

        # Показать информацию о производительности
        print("\n🚀 Производительность:")
        print(
            f"   Оптимизировано для batch_size={args.opt_batch or cfg.training.training.batch_size}"
        )
        print(
            f"   Точность: {args.precision or cfg.training.production.tensorrt.precision}"
        )
        print(
            f"   Входной формат: [{args.opt_batch or cfg.training.training.batch_size}, {cfg.training.data.sequence_length}, {cfg.training.data.input_size_per_frame}]"
        )

        return 0
    else:
        print("\n❌ Конвертация не удалась")
        print("💡 Проверьте логи выше и исправьте ошибки")
        return 1


if __name__ == "__main__":
    sys.exit(main())
