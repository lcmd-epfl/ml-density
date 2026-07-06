"""Logging configuration."""

import os
import sys
import logging
import copy
from libs.config import DEFAULT_LOGLEVEL


class MultilineMixin:
    '''https://docs.python.org/3/howto/logging-cookbook.html#how-to-uniformly-handle-newlines-in-logging-output'''
    def emit(self, record):
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
    pass


class StreamFlushingHandler(MultilineMixin, logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(name, caller, level=DEFAULT_LOGLEVEL):
    """Configure logger.

    Args:
        name (str): Logger name (typically __name__).
        caller (str): Path to the calling script.
        level: Logging level (default: INFO).  Note: will be overwritten
               by the CLI arguments parser (see libs/config.py).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    def only_flush(record):
        return getattr(record, 'flush', None)

    formatter = logging.Formatter(
            f'%(asctime)s {os.path.basename(caller)} [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            )

    # normal handler
    handler = StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(lambda x: not only_flush(x))
    logger.addHandler(handler)

    # handler that flushes
    handler_flushing = StreamFlushingHandler(sys.stderr)
    handler_flushing.addFilter(only_flush)
    handler_flushing.setFormatter(formatter)
    logger.addHandler(handler_flushing)

    return logger
