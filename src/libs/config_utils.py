"""Utilities for configuration and CLI parsing."""

import logging
from types import SimpleNamespace
from typing import NamedTuple
from enum import Enum
from collections.abc import Callable
import numpy as np

logger = logging.getLogger('__main__')


defaults = SimpleNamespace(config='config.txt', mpi=True, loglevel=logging.DEBUG)


class WhenMissing(Enum):
    """Policy for handling missing configuration entries."""
    IGNORE = 0
    WARN = 1
    ERROR = 2


class CheckFile(Enum):
    """Validation policy applied to configured paths."""
    ERROR_FILE = 1
    ERROR_DIR = 2
    MAKE_DIR = 3


class PathSpecs(NamedTuple):
    """Specification of one path option in a config section."""
    key: str
    when_missing: WhenMissing
    check_file: CheckFile
    default: str | None


class OptSpecs(NamedTuple):
    """Specification of one variable option in a config section."""
    key: str                             # Option key inside the config section.
    default: object                      # Default value when the key is missing.
    dtype: Callable[[str], object]       # Callable converting raw string values.


class Floats:
    """Parser converting comma-separated strings to unique float arrays."""
    def __call__(self, x):
        """Parse a comma-separated list of floats.

        Args:
            x (str): Raw input value.

        Returns:
            np.ndarray: Sorted unique float values.
        """
        return np.unique(list(map(float, x.split(','))))


class Bool:
    """Parser converting common textual flags to booleans."""
    def __call__(self, x):
        """Parse a textual boolean option.

        Args:
            x (str): Raw input value.

        Returns:
            bool: Parsed and validated value.

        Raises:
            TypeError: x cannot be interpreted as boolean value.
        """
        x = x.lower()
        if x in {'1', 'true', 'on', 'yes'}:
            return True
        if x in {'0', 'false', 'off', 'no'}:
            return False
        msg = f'Wrong input for a Bool option: "{x}"'
        raise TypeError(msg)


class Choice:
    """Parser wrapper enforcing membership in an allowed option set."""

    def __init__(self, dtype, options, name, *, strict=True):
        """Initialize a Choice instance.

        Args:
            dtype (Callable[[str], T]): Converter used to cast raw configuration values.
            options (list[T]): Accepted options (after casting).
            name (str): Option name (for logging/error messages).
            strict (bool): Whether invalid values should raise an error.
        """
        self.dtype = dtype
        self.options = options
        self.name = name
        self.strict = strict

    def __call__(self, x):
        """Cast and validate a value against the configured choices.

        Args:
            x (str): Raw input value.

        Returns:
            object: Validated value cast with self.dtype.

        Raises:
            RuntimeError: Value is not in accepted options and the validation is strict.
        """
        if (x := self.dtype(x)) not in self.options:
            if self.strict:
                msg = f'Wrong input for `{self.name}`: `{x}` not in {self.options}'
                raise RuntimeError(msg)
            logger.warning(f'`{x}` not in the recommended options {self.options} for `{self.name}`')
        return x
