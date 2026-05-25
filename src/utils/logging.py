"""Logging and experiment tracking utilities."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "speaker_id",
    log_dir: str = "logs",
    experiment_id: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger that writes to both stdout and a file.

    Args:
        name: Logger name.
        log_dir: Base directory for log files.
        experiment_id: Optional experiment identifier. If None, a timestamp-based
            ID is generated.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(log_dir) / experiment_id
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{name}.{experiment_id}")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path / "experiment.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_experiment_config(
    config: dict,
    log_dir: str = "logs",
    experiment_id: str | None = None,
) -> Path:
    """Save experiment configuration as JSON.

    Args:
        config: Experiment configuration dictionary.
        log_dir: Base directory for log files.
        experiment_id: Optional experiment identifier.

    Returns:
        Path to the saved config file.
    """
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(log_dir) / experiment_id
    log_path.mkdir(parents=True, exist_ok=True)

    config_path = log_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return config_path


def save_metrics(
    metrics: dict,
    log_dir: str = "logs",
    experiment_id: str | None = None,
) -> Path:
    """Save experiment metrics as JSON.

    Args:
        metrics: Metrics dictionary.
        log_dir: Base directory for log files.
        experiment_id: Optional experiment identifier.

    Returns:
        Path to the saved metrics file.
    """
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(log_dir) / experiment_id
    log_path.mkdir(parents=True, exist_ok=True)

    metrics_path = log_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics_path


def append_metrics_jsonl(
    metrics: dict,
    log_dir: str = "logs",
    experiment_id: str | None = None,
) -> Path:
    """Append a metrics record to a JSONL file for incremental tracking.

    Args:
        metrics: Metrics dictionary (one record).
        log_dir: Base directory for log files.
        experiment_id: Optional experiment identifier.

    Returns:
        Path to the JSONL file.
    """
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(log_dir) / experiment_id
    log_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = log_path / "metrics.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(metrics, default=str) + "\n")

    return jsonl_path
