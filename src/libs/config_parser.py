"""Configuration parsing primitives."""

import os
import configparser
import logging
from libs.config_utils import WhenMissing, CheckFile, defaults

logger = logging.getLogger('__main__')


class Config:
    """Read, validate, and parse configuration sections."""

    def __init__(self, config_path=defaults.config):
        """Initialize a Config instance.

        Args:
            config_path (str | None): Path to the configuration file.

        Raises:
            RuntimeError: When the file is not found.
        """
        if not os.path.isfile(config_path):
            msg = f'Cannot open configuration file "{config_path}"'
            raise RuntimeError(msg)
        link = f' -> {os.readlink(config_path)}' if os.path.islink(config_path) else ''
        logger.info(f'Configuration file: {config_path}{link}')
        parser = configparser.RawConfigParser()
        parser.read(config_path)
        self.configuration = dict(parser.items())

    def get_option(self, group, specs):
        """Read and cast one option from a configuration section.

        Args:
            group (str): Configuration section name.
            specs (OptSpecs): Parsing specs.

        Returns:
            object: Parsed option value.
        """
        return specs.dtype(val) if (val := self.configuration[group].get(specs.key, None)) is not None else specs.default

    def check_for_unrecognized_options(self, group, options):
        """Check for unrecognized options.

        Args:
            group (str): Configuration section name.
            options (dict[str, .config_utils.PathSpecs | .config_utils.OptSpecs]): Mapping from destination names to parsing specs.

        Raises:
            RuntimeError: There are unrecognized options in the config section.
        """
        recognized = [val.key for val in options.values()]
        present = self.configuration[group].keys()
        if unrecognized := set(present).difference(recognized):
            msg = f'Unrecognized entries in [{group}]: {unrecognized}'
            raise RuntimeError(msg)

    def get_option_group(self, group, options):
        """Read all options declared for one configuration section.

        Args:
            group (str): Configuration section name.
            options (dict[str, .config_utils.OptSpecs]): Mapping from destination names to parsing specs.

        Returns:
            dict[str, object]: Parsed options keyed by destination name.
        """
        self.check_for_unrecognized_options(group, options)
        return {dest: self.get_option(group, specs) for dest, specs in options.items()}

    def get_path_group(self, group, paths):
        """Read and validate all path entries for one section.

        Create empty directories if they are specified and missing.

        Args:
            group (str): Configuration section name.
            paths (dict[str, .config_utils.PathSpecs]): Mapping from destination names to parsing specs.

        Returns:
            dict[str, str]: Resolved path values keyed by destination name.

        Raises:
            RuntimeError: A required file does not exist.
            RuntimeError: A required directory does not exist.
        """
        self.check_for_unrecognized_options(group, paths)
        p = {}
        for dest, spec in paths.items():
            if (path := self.configuration[group].get(spec.key)) is None:
                if spec.when_missing==WhenMissing.WARN:
                    logger.warning(f'Missing recommended config path `{spec.key}`')
                elif spec.when_missing==WhenMissing.ERROR:
                    msg = f'Missing required config path `{spec.key}`'
                    raise RuntimeError(msg)
                else:
                    path = spec.default

            elif spec.check_file==CheckFile.ERROR_FILE:
                if not os.path.isfile(path):
                    msg = f'Missing file "{path}" ("{spec.key}")'
                    raise RuntimeError(msg)
            elif spec.check_file in {CheckFile.ERROR_DIR, CheckFile.MAKE_DIR}:
                fdir = os.path.dirname(path)
                if not os.path.isdir(fdir):
                    if spec.check_file==CheckFile.ERROR_DIR:
                        msg = f'Missing directory "{fdir}" ("{spec.key}")'
                        raise RuntimeError(msg)
                    logger.debug(f'Creating directory "{fdir}" ("{spec.key}")')
                    os.makedirs(fdir)
            p[dest] = path
        return p
