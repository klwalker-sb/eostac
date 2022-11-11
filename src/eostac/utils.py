import logging

from joblib import Parallel
from tqdm.auto import tqdm


class TqdmParallel(Parallel):
    """A tqdm progress bar for joblib Parallel tasks
    Reference:
        https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """
    def __init__(self, tqdm_kwargs: dict, **joblib_kwargs):
        self.tqdm_kwargs = tqdm_kwargs
        super().__init__(**joblib_kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(**self.tqdm_kwargs) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class ColorFormatter(logging.Formatter):
    """Reference:
        https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s [%(levelname)s] [%(name)s.py:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d:%H:%M:%S')

        return formatter.format(record)


def set_color_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    return logger
