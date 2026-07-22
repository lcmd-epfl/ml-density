"""Configuration parsing primitives."""

import os
import configparser
import logging
from libs.config_utils import WhenMissing, CheckFile, OptSpecs, PathSpecs, Choice, mpi_rank

logger = logging.getLogger('__main__')


class Config:
    """Read, validate, and parse configuration sections."""

    def __init__(self, config_groups=None):
        """Initialize a Config instance.

        Args:
            config_groups(dict[str, dict[str, OptSpecs | PathSpecs]]): Specification of option/path groups and entries.
        """
        self.groups = {}
        if config_groups is not None:
            for group, entries in config_groups.items():
                self.add_group(group)
                for dest, spec in entries.items():
                    self.add_entry(group, dest, spec)

    def add_group(self, group):
        """Register a new parsing group.

        Args:
            group (str): Group/section identifier used by this parser.

        Raises:
            RuntimeError: The group is already registered.
        """
        if group in self.groups:
            msg = f'Duplicated group: {group}'
            raise RuntimeError(msg)
        self.groups[group] = {}

    def add_entry(self, group, key, spec):
        """Register one option/path specification under a group.

        Args:
            group (str): Previously registered group name.
            key (str): Destination key used in parsed output.
            spec (OptSpecs | PathSpecs): Specification describing how to parse the entry.

        Raises:
            RuntimeError: The group does not exist.
            RuntimeError: The key is already registered in the group.
        """
        if group not in self.groups:
            msg = f'Non-existing group: {group}'
            raise RuntimeError(msg)
        if key in self.groups[group]:
            msg = f'Duplicated key: {key}'
            raise RuntimeError(msg)
        self.groups[group][key] = spec

    def parse(self, config_path):
        """Parse all registered groups.

        Args:
            config_path (str | None): Path to the configuration file.

        Returns:
            dict[str, dict[str, object]]: Parsed values grouped by group name.

        Raises:
            RuntimeError: When the file is not found.
        """
        if not os.path.isfile(config_path):
            msg = f'Cannot open configuration file "{config_path}"'
            raise RuntimeError(msg)
        link = f' -> {os.readlink(config_path)}' if os.path.islink(config_path) else ''
        if mpi_rank() == 0:
            logger.info(f'Configuration file: {config_path}{link}')

        parser = configparser.RawConfigParser()
        parser.read(config_path)
        self.configuration = dict(parser.items())

        return {group : self.parse_group(group, options) for group, options in self.groups.items()}

    def print_help(self):
        """Print an example configuration file generated from registered specs."""
        def get_value(spec):
            if spec.default is None:
                return '<required>'
            if (dtype:=getattr(spec, 'dtype', None)) and (str_:=getattr(dtype, 'str', None)):
                return str_(spec.default)
            return str(spec.default)

        def get_help(spec):
            descr = getattr(spec, 'help', '')
            if (dtype:=getattr(spec, 'dtype', None)) and isinstance(dtype, Choice):
                descr += (f' {'Permitted' if dtype.strict else 'Recommended'} values: {dtype.options}.')
            return f'  # {descr.strip()}' if descr else ''

        lines = ['# Example configuration file with default values.\n']
        for group, entries in self.groups.items():
            lines.append(f'[{group}]')
            lines_group = []
            for spec in entries.values():
                keyval = f'{spec.key} = {get_value(spec)}'
                descr = get_help(spec)
                lines_group.append((keyval, descr))

            maxlen = max(len(keyval) for keyval, _ in lines_group)
            lines.extend([keyval + ' '*(maxlen-len(keyval)) + descr for keyval, descr in lines_group])
            lines.append('')
        print('\n'.join(lines).rstrip() + '\n')

    def parse_entry(self, group, spec):
        """Parse and resolve one configuration entry.

        Args:
            group (str): Configuration section name.
            spec (OptSpecs | PathSpecs): Parsing spec.

        Returns:
            object: Parsed option value or resolved path.

        Raises:
            TypeError: The spec type is unsupported.
        """
        if isinstance(spec, OptSpecs):
            return self._get_option(group, spec)
        if isinstance(spec, PathSpecs):
            return self._get_path(group, spec)
        msg = f'Wrong type of option specification ({type(spec)})'
        raise TypeError(msg)

    def _get_option(self, group, spec):
        """Parse one option using its converter and default.

        Args:
            group (str): Configuration section name.
            spec (OptSpecs): Option parsing specification.

        Returns:
            object: Parsed value, or the default when missing.
        """
        return spec.dtype(val) if (val := self.configuration[group].get(spec.key, None)) is not None else spec.default

    def check_for_unrecognized_entries(self, group, entries):
        """Check for unrecognized entries in the config file.

        Args:
            group (str): Configuration section name.
            entries (dict[str, PathSpecs | OptSpecs]): Mapping from destination names to parsing specs.

        Raises:
            RuntimeError: The config section contains undeclared keys.
        """
        recognized = [val.key for val in entries.values()]
        present = self.configuration[group].keys()
        if unrecognized := set(present).difference(recognized):
            msg = f'Unrecognized entries in [{group}]: {unrecognized}'
            raise RuntimeError(msg)

    def parse_group(self, group, entries):
        """Read all entries declared for one configuration section.

        Args:
            group (str): Configuration section name.
            entries (dict[str, PathSpecs | OptSpecs]): Mapping from destination names to parsing specs.

        Returns:
            dict[str, object]: Parsed entries keyed by destination name.
        """
        self.check_for_unrecognized_entries(group, entries)
        return {dest: self.parse_entry(group, specs) for dest, specs in entries.items()}

    def _get_path(self, group, spec):
        """Parse and validate one path entry.

        Missing parent directories are created when requested by the spec.

        Args:
            group (str): Configuration section name.
            spec (PathSpecs): Path parsing specification.

        Returns:
            str | None: Resolved path value (or None when allowed by the specification).

        Raises:
            RuntimeError: A required path entry is missing.
            RuntimeError: A required file does not exist.
            RuntimeError: A required directory does not exist.
        """
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
        return path
