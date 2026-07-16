"""CLI parsing for configuration-driven scripts."""

import argparse
import logging
from libs.config_utils import defaults


def parse_cli_args(return_args=None):
    """Parse supported command-line arguments.

    Args:
        return_args (list[str] | None): Optional whitelist of extra arguments to expose.

    Returns:
        argparse.Namespace: Parsed CLI arguments namespace.
    """
    if return_args is None:
        return_args = []

    def add_argument(parser, *kargs, **kwargs):
        """Conditionally register an argument only when requested by return_args.

        Args:
            parser (argparse.ArgumentParser): Parser that may receive the argument.
            *kargs (object): Positional arguments forwarded to add_argument.
            **kwargs (object): Keyword arguments forwarded to add_argument.
        """
        if kwargs.get('dest') in return_args:
            parser.add_argument(*kargs, **kwargs)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default=defaults.config, help='path to the configuration file')
    parser.add_argument("--help-config", "--config-help", dest='config_help', action='store_true', help='print configuration file help')
    parser.add_argument("--log", type=str, default=logging._levelToName[defaults.loglevel], choices=logging._nameToLevel.keys(), help='logging level')
    add_argument(parser, "-b", "--b", dest="get_b_matrix", action='store_true', help='if True, get_matrices computes the "B matrix"; if False, the "A vector"')
    add_argument(parser, "--missing-only", dest="missing_only", action='store_true', help='dangerous: not recompute existing power spectra / kernels')

    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--subset-dummy-argument", action='store_true', help=argparse.SUPPRESS)
    add_argument(subset, "--training", "--train", dest='training', action='store_true', help='run prediction / compute error on the training set instead of the test set')
    add_argument(subset, "--extra", "--extrapolation", "--oos", dest="extra", action='store_true', help='run script for an out-of-sample set (extrapolation)')

    mpi = parser.add_mutually_exclusive_group()
    mpi.add_argument("--mpi-dummy-argument", dest="mpi", default=defaults.mpi, action='store_true', help=argparse.SUPPRESS)
    add_argument(mpi, "--mpi", dest="mpi", default=defaults.mpi, action='store_true', help='set MPI usage flag')
    add_argument(mpi, "--no-mpi", dest="mpi", default=defaults.mpi, action='store_false', help='set MPI usage flag')
    return parser.parse_args()
