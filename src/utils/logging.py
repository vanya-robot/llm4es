"""
Логирование и таймеры.
"""
import logging
import sys
import time
from contextlib import contextmanager

_logger = None


def get_logger(name: str = "llm4es", level: int = logging.INFO) -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    _logger = logger
    return logger


@contextmanager
def timer(description: str, logger: logging.Logger = None):
    if logger is None:
        logger = get_logger()
    logger.info(f"Starting: {description}")
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"Finished: {description} ({elapsed:.2f}s)")


def log_memory(logger: logging.Logger = None):
    if logger is None:
        logger = get_logger()
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info().rss / 1024 ** 2
        logger.info(f"RSS memory: {mem:.1f} MB")
    except ImportError:
        logger.debug("psutil not installed, skipping memory log")
