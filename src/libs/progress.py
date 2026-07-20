"""tqdm wrappers that fall silent when output is not a terminal (e.g. slurm logs).

tqdm redraws its bar with carriage returns, which turns into one unreadable line when
stderr is redirected to a file. `disable=None` is tqdm's "auto-disable on non-TTY" mode:
interactive runs keep the live bar, batch logs get nothing.
"""

from functools import partial
import tqdm as _tqdm

tqdm = partial(_tqdm.tqdm, disable=None)
trange = partial(_tqdm.trange, disable=None)
