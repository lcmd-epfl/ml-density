"""Utilities for configuration and CLI parsing."""

import os
import logging
from types import SimpleNamespace
from typing import NamedTuple
from enum import Enum
from collections.abc import Callable
import numpy as np

logger = logging.getLogger('__main__')


defaults = SimpleNamespace(config='config.txt', mpi=True, loglevel=logging.INFO, default_dir='INNER/')

DEFAULT_DIR_GROUP = 'paths'
DEFAULT_DIR_KEY = 'default_dir'
DEFAULT_DIR_PLACEHOLDER = f'{{{DEFAULT_DIR_KEY}}}'  # '{default_dir}'


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


class Sparsification(Choice):
    """Parser selecting the sparse-GPR approximation: `false | pitc | dtc`.

    Parsed to the union `False | 'pitc' | 'dtc'` rather than to a plain string, so that the many
    `if o.full_gpr:` sites throughout the pipeline keep meaning "is this a full-GPR run?": `False`
    is falsy while both method names are truthy. A plain `Choice(str, ['false', 'pitc', 'dtc'])`
    would return the *string* `'false'`, which is truthy, and would silently turn every one of
    those guards on.

    The former boolean spelling `true` is rejected rather than aliased to `pitc`: `full_gpr = true`
    used to mean PITC, and silently keeping that meaning would hide the choice this option now
    carries.
    """
    __name__ = 'Sparsification'

    OFF = 'false'
    TRUTHY = frozenset({'1', 'true', 'on', 'yes'})

    def __init__(self, name):
        """Initialize a Sparsification instance.

        Args:
            name (str): Option name (for logging/error messages).
        """
        super().__init__(str, [self.OFF, 'pitc', 'dtc'], name)

    def __call__(self, x):
        """Parse the sparsification method.

        Args:
            x (str): Raw input value.

        Returns:
            bool | str: `False` for the SoR path, otherwise `'pitc'` or `'dtc'`.

        Raises:
            RuntimeError: Value is a boolean truthy spelling, or not one of the accepted options.
        """
        if (x := x.strip().lower()) in self.TRUTHY:
            msg = (f'`{self.name} = {x}` is no longer accepted: it used to select PITC, but PITC is '
                   f'now one of several sparsifications. Write `{self.name} = pitc` for the previous '
                   f'behaviour, or `{self.name} = dtc`.')
            raise RuntimeError(msg)
        return False if (x := super().__call__(x)) == self.OFF else x

    def str(self, x):
        """Render a parsed value back to its configuration spelling.

        Args:
            x (bool | str): Parsed value.

        Returns:
            str: The spelling accepted by __call__.
        """
        return self.OFF if x is False else x
