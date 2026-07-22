"""Utilities for configuration and CLI parsing."""

import os
import logging
from types import SimpleNamespace
from typing import NamedTuple
from enum import Enum
from collections.abc import Callable
import numpy as np

logger = logging.getLogger('__main__')


defaults = SimpleNamespace(config='config.txt', mpi=True, loglevel=logging.INFO)


def mpi_rank():
    """Best-effort MPI rank from launcher environment variables.

    Reads the rank exported by the job launcher (``srun`` or ``mpirun``) without
    importing ``mpi4py`` (which would trigger ``MPI_Init``). Used to keep rank-0-only
    log lines from being duplicated across ranks that share one stdout.

    Returns:
        int: The current MPI rank, or 0 when not running under a recognised launcher.
    """
    for var in ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK', 'SLURM_PROCID'):
        if (val := os.environ.get(var)) is not None:
            return int(val)
    return 0


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
    help: str


class OptSpecs(NamedTuple):
    """Specification of one variable option in a config section."""
    key: str                             # Option key inside the config section.
    default: object                      # Default value when the key is missing.
    dtype: Callable[[str], object]       # Callable converting raw string values.
    help: str


class Floats:
    """Parser converting comma-separated strings to unique float arrays."""
    __name__ = 'Floats'

    def __call__(self, x):
        """Parse a comma-separated list of floats.

        Args:
            x (str): Raw input value.

        Returns:
            np.ndarray: Sorted unique float values.
        """
        return np.unique(list(map(float, x.split(','))))

    def str(self, x):
        """Convert floats to a comma-separated list.

        Args:
            x (np.ndarray[float] | list[float]): list to convert.

        Returns:
            str: Comma-separated list.
        """
        return ','.join(map(str, x))


class Bool:
    """Parser converting common textual flags to booleans."""
    __name__ = 'Bool'

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
    __name__ = 'Choice'

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

    def __repr__(self):
        return f'{self.__name__}({self.dtype.__name__}, {self.options}, name="{self.name}", strict={self.strict})'

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
