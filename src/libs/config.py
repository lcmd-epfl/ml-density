"""Parse CLI arguments and a configuration file."""

import logging
from libs.config_paths import read_config
from libs.config_cli import parse_cli_args

logger = logging.getLogger('__main__')


def get_settings(return_args=None):
    """Load CLI args and configuration options/paths for a script.

    Args:
        return_args (list[str] | None): Optional list of script-specific CLI flags to parse.

    Returns:
        tuple: Either (options, paths) or (args, options, paths) depending on return_args.
    """
    args = parse_cli_args(return_args)
    logger.setLevel(args.log)
    o, p = read_config(args.config, print_help=args.config_help)
    return (args, o, p) if return_args else (o, p)
