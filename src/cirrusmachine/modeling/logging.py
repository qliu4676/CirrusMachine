"""Opt-in console logging and coarse optimization progress."""

import logging
import math
import os
import sys

from tqdm.auto import tqdm


class _Formatter(logging.Formatter):
    colors = {
        logging.DEBUG: 36,
        logging.INFO: 32,
        logging.WARNING: 33,
        logging.ERROR: 31,
        logging.CRITICAL: 35,
    }

    def __init__(self, color):
        super().__init__("[%(levelname)s] %(message)s")
        self.color = color

    def format(self, record):
        message = super().format(record)
        if self.color:
            label = f"[{record.levelname}]"
            return (
                f"\033[{self.colors.get(record.levelno, 0)}m{label}\033[0m"
                + message[len(label) :]
            )
        return message


class _Handler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def configure_logging(level="INFO", *, color="auto", stream=None):
    """Configure only CirrusMachine logging; safe to rerun in a notebook.

    Levels follow Python logging (DEBUG, INFO, WARNING, ERROR). Automatic color
    supports terminals and Jupyter, respects NO_COLOR, and is off in log files.
    Only the level label is colored; message text keeps the default color.
    Pass color=False for plain output or color=True to force ANSI colors.
    """
    if color not in (True, False, "auto"):
        raise ValueError("color must be True, False or 'auto'")
    stream = sys.stderr if stream is None else stream
    if color == "auto":
        color = "NO_COLOR" not in os.environ and (
            stream.isatty() or stream.__class__.__module__.startswith("ipykernel")
        )
    logger = logging.getLogger("cirrusmachine")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if isinstance(handler, _Handler):
            logger.removeHandler(handler)
            handler.close()
    handler = _Handler(stream)
    handler.setFormatter(_Formatter(color))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class ModelingProgress:
    """Refresh once per tenth of a draw, with guaranteed cleanup on failure."""

    def __init__(self, steps, draw, draws, progress):
        self.steps, self.draw = steps, draw
        self.callback = progress if callable(progress) else None
        self.enabled = progress is not False
        self.milestones = {math.ceil(steps * tenth / 10) for tenth in range(1, 11)}
        self.bar = None
        self.description = f"Draw {draw + 1}/{draws}"

    def __enter__(self):
        if self.enabled and self.callback is None:
            self.bar = tqdm(
                total=self.steps,
                desc=self.description,
                unit="step",
                mininterval=0,
                miniters=1,
                dynamic_ncols=True,
            )
        return self

    def update(self, step, loss, learning_rate):
        if step not in self.milestones:
            return
        if self.bar is not None:
            self.bar.set_postfix(
                loss=f"{loss:.4g}", lr=f"{learning_rate:g}", refresh=False
            )
            self.bar.update(step - self.bar.n)
        if self.callback:
            self.callback(
                dict(draw=self.draw, step=step, loss=loss, learning_rate=learning_rate)
            )

    def __exit__(self, *exc):
        if self.bar is not None:
            self.bar.close()
