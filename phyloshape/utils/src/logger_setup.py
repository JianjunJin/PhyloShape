#!/usr/bin/env python

"""Logger for development and user warnings.
All phyloshape modules that use logging use a 'bound' logger that will
be filtered here to only show for phyloshape and not for other Python
packages.
"""

import sys
from loguru import logger
import phyloshape


def colorize():
    """colorize the logger if stderr is IPython/Jupyter or a terminal (TTY)"""
    try:
        import IPython        
        tty1 = bool(IPython.get_ipython())
    except ImportError:
        tty1 = False
    tty2 = sys.stderr.isatty()
    if tty1 or tty2:
        return True
    return False


LOGGERS = [0]


def set_log_level(log_level="INFO"):
    """Set the log level for loguru logger bound to phyloshape.
    This removes default loguru handler, but leaves any others in place, 
    and adds a new one that will filter to only print logs from 
    phyloshape modules, which should use `logger.bind(name='phyloshape')`.
    Examples
    --------
    >>> # suppress phyloshape logs below INFO level
    >>> phyloshape.set_log_level("INFO") 
    >>>
    >>> # write a log message from the phyloshape logger
    >>> from loguru import logger
    >>> logger.bind(name="phyloshape").info("logged message from phyloshape")
    """
    for idx in LOGGERS:
        try:
            logger.remove(idx)
        except ValueError:
            pass

    if log_level in ("DEBUG", "TRACE"):
        idx = logger.add(
            sink=sys.stderr,
            level=log_level,
            colorize=colorize(),
            format="{level.icon} {time:YYYY-MM-DD-HH:mm:ss.SS} | "
                   "<magenta>{file: >15} | </magenta>"
                   "<cyan>{function: <25} | </cyan>"
                   "<level>{message}</level>",
            filter=lambda x: x['extra'].get("name") == "phyloshape",
        )
    else:
        idx = logger.add(
            sink=sys.stderr,
            level=log_level,
            colorize=colorize(),
            format="{level.icon} {time:YYYY-MM-DD-HH:mm:ss.SS} | "
                   "<level>{message}</level>",
            filter=lambda x: x['extra'].get("name") == "phyloshape",
        )
    LOGGERS.append(idx)
    logger.enable("phyloshape")
    logger.bind(name="phyloshape").debug(
        f"phyloshape v.{phyloshape.__version__} logging enabled"
    )