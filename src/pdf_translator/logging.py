"""Structured logging via a shared rich console.

All stages log through ``get_logger(__name__)``. The CLI calls
:func:`setup_logging` once to attach a :class:`~rich.logging.RichHandler` and to
expose the shared :data:`console` used for progress bars and live metrics.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

# One console shared by logging, progress bars, and live metrics so output never
# interleaves badly.
console = Console(stderr=True)

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once. ``verbose`` enables DEBUG + source paths."""
    global _CONFIGURED
    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        show_path=verbose,
        markup=True,
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party loggers unless the user asked for verbose output.
    if not verbose:
        for noisy in ("httpx", "httpcore", "pdfminer", "PIL", "fontTools", "weasyprint"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging lazily if the CLI hasn't yet."""
    if not _CONFIGURED:
        setup_logging(False)
    return logging.getLogger(name)
