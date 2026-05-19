import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured = False


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path / "medtender.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)

    root_logger = logging.getLogger("medtender")
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"medtender.{name}")
