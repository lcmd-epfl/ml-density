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


FIT_REG = 'fit'


class FloatOrFit:
    """Parser for a hyperparameter that is either pinned to a positive float or fitted from data.

    Used for both hyperparameters of main.pdf Sec. IIF -- `regularisation` (lambda, Eq. 17) and
    `prior_scale` (sigma_p^2, Eq. 26). `fit` is represented as None throughout, so that any code
    reading the option as a number fails loudly instead of silently using a placeholder: the fitted
    value only exists once training has run, and is read back from the file regression.py writes.
    """
    __name__ = 'FloatOrFit'

    def __init__(self, name):
        """Initialize a FloatOrFit instance.

        Args:
            name (str): Option name, used in error messages.
        """
        self.name = name

    def __repr__(self):
        return f'{self.__name__}(name="{self.name}")'

    def __call__(self, x):
        """Parse the option value.

        Args:
            x (str): Raw input value.

        Returns:
            float | None: The pinned value, or None when it is to be fitted.

        Raises:
            RuntimeError: The value is neither `fit` nor a positive float.
        """
        if str(x).strip().lower() == FIT_REG:
            return None
        try:
            value = float(x)
        except ValueError:
            msg = f'Wrong input for `{self.name}`: "{x}" is neither a float nor "{FIT_REG}"'
            raise RuntimeError(msg) from None
        if value <= 0.0:
            msg = f'`{self.name}` must be positive, got {value}'
            raise RuntimeError(msg)
        return value

    def str(self, x):
        """Render a parsed value back into config syntax.

        Args:
            x (float | None): Parsed value.

        Returns:
            str: The config representation.
        """
        return FIT_REG if x is None else str(x)


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


SAGPR = 'sagpr'
GPR_DTC = 'gpr_DTC'
GPR_PITC = 'gpr_PITC'
GPR_MODELS = (GPR_DTC, GPR_PITC)


class RegressionModel(Choice):
    """Parser selecting the regression model: `sagpr | gpr_DTC | gpr_PITC`.

    - `sagpr`    the deterministic symmetry-adapted GPR fit, which yields a prediction without uncertainty;
                 Grisafi et al. (2019) & Fabrizio et al. (2019)
    - `gpr_DTC`  the Deterministic Training Conditional sparse GP, 
                 which leads to the same prediction but adds a posterior;
    - `gpr_PITC` the Partially Independent Training Conditional sparse GP, which additionally
                 carries the per-molecule Nystrom residual D_i=K_ii - K_iM K_MM K_Mi.

    Values are matched case-insensitively.

    The retired spellings of the former `full_gpr` option are rejected rather than aliased.
    """
    __name__ = 'RegressionModel'

    # Retired `full_gpr` spellings -> the model they used to select.
    RETIRED = {'1': GPR_PITC, 'true': GPR_PITC, 'on': GPR_PITC, 'yes': GPR_PITC, 'pitc': GPR_PITC,
               '0': SAGPR, 'false': SAGPR, 'off': SAGPR, 'no': SAGPR, 'sor': SAGPR, 'dtc': GPR_DTC}

    def __init__(self, name):
        """Initialize a RegressionModel instance.

        Args:
            name (str): Option name (for logging/error messages).
        """
        super().__init__(str, [SAGPR, GPR_DTC, GPR_PITC], name)
        self._canonical = {value.lower(): value for value in self.options}

    def __call__(self, x):
        """Parse the regression model.

        Args:
            x (str): Raw input value, matched case-insensitively.

        Returns:
            str: One of SAGPR, GPR_DTC, GPR_PITC.

        Raises:
            RuntimeError: Value is a retired `full_gpr` spelling, or not one of the accepted options.
        """
        x = x.strip().lower()
        if x not in self._canonical and x in self.RETIRED:
            msg = (f'`{self.name} = {x}` is a retired spelling of the old `full_gpr` option. '
                   f'Write `{self.name} = {self.RETIRED[x]}` instead (accepted values: '
                   f'{self.options}).')
            raise RuntimeError(msg)
        return super().__call__(self._canonical.get(x, x))
