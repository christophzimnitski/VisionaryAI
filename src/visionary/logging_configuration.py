"""Project-wide logging setup.

Replaces the scattered ``print`` calls in the original scripts with a single
configurable logger so tracking events (new track, lost, recovered, collision)
become structured, level-filtered log records.
"""

from __future__ import annotations

import logging

_LOG_MESSAGE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_root_logging(logging_level: int = logging.INFO) -> None:
    """Configure the root logger once, for use by command-line entry points."""
    logging.basicConfig(level=logging_level, format=_LOG_MESSAGE_FORMAT)


def get_logger_for_module(module_name: str) -> logging.Logger:
    """Return a named logger for the given module."""
    return logging.getLogger(module_name)
