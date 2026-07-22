"""Logging configuration."""

import os
import sys
import logging
import copy
from libs.config_utils import defaults


class MultilineMixin:
    """Mixin splitting multi-line log messages into single-line records.

    Taken from
    https://docs.python.org/3/howto/logging-cookbook.html#how-to-uniformly-handle-newlines-in-logging-output
    """
    def emit(self, record):
        """Emit a log record, splitting multi-line messages into separate records.

        Args:
            record (logging.LogRecord): Log record emitted by the handler.
        """
        s = record.getMessage()
        if '\n' not in s:
            super().emit(record)
        else:
            lines = s.splitlines()
            rec = copy.copy(record)
            rec.args = None
            for line in lines:
                rec.msg = line
                super().emit(rec)


class StreamHandler(MultilineMixin, logging.StreamHandler):
    """Stream handler with multi-line aware emit behavior."""


class StreamFlushingHandler(MultilineMixin, logging.StreamHandler):
    """Stream handler that flushes immediately after each emitted record."""
    def emit(self, record):
        """Emit a record and flush the stream immediately.

        Args:
            record (logging.LogRecord): Log record emitted by the handler.
        """
        super().emit(record)
        self.flush()


def setup_logger(name, caller, level=defaults.loglevel):
    """Configure project logging with normal and flushing stream handlers.

    Args:
        name (str): Logger name (typically __name__).
        caller (str): Path to the calling script.
        level (int): Logging level (default: INFO). Note: this is usually overwritten
                     by CLI arguments parsing.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    def only_flush(record):
        """Select records that request immediate flushing.

        Args:
            record (logging.LogRecord): Record to inspect.

        Returns:
            bool: The `flush` attribute of the record if it exists, otherwise False.
        """
        return getattr(record, 'flush', False)

    formatter = logging.Formatter(
            f'%(asctime)s {os.path.basename(caller)} [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            )

    # normal handler. Deliberately left at NOTSET so that it inherits the logger level.
    handler = StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(lambda x: not only_flush(x))
    logger.addHandler(handler)

    # handler that flushes
    handler_flushing = StreamFlushingHandler(sys.stderr)
    handler_flushing.addFilter(only_flush)
    handler_flushing.setFormatter(formatter)
    logger.addHandler(handler_flushing)

    return logger
