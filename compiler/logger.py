import logging
from typing import List, Set, Tuple, Optional

LOG = logging.getLogger('evfl-compiler')
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    formatter = logging.Formatter(f"%(levelname)8s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    LOG.addHandler(stream_handler)

logging.addLevelName(logging.FATAL, 'fatal')
logging.addLevelName(logging.ERROR, 'error')
logging.addLevelName(logging.WARNING, 'warning')
logging.addLevelName(logging.INFO, 'info')
logging.addLevelName(logging.DEBUG, 'debug')

_sent_messages: Set[str] = set()
_file_contents: List[str] = []

def set_log_level(level) -> None:
    LOG.setLevel(level)
    stream_handler.setLevel(level)

def init_logger(filename) -> None:
    global _file_contents, _sent_messages
    _sent_messages = set()
    formatter = logging.Formatter(f"{filename}: %(levelname)8s: %(message)s")
    stream_handler.setFormatter(formatter)
    _file_contents = []

def setup_logger(filecontent) -> None:
    global _file_contents
    _file_contents = filecontent.replace('\r', '').split('\n')

def log(level, message: str,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None,
        print_source: bool = True) -> None:
    if not end:
        end = start
    if message not in _sent_messages:
        _sent_messages.add(message)
        if start:
            message = f'{start[0]}:{start[1]}: {message}'
        else:
            message = f' {message}'
        LOG.log(level, message)
        if print_source and start and  1 <= start[0] <= len(_file_contents) and 1 <= start[1] <= len(_file_contents[start[0] - 1]):
            assert end is not None
            num_markers = min(
                len(_file_contents[start[0] - 1]) - (start[1] - 1),
                end[1] - start[1] + 1 if end[0] == start[0] else 999999,
            )
            LOG.log(logging.INFO, f'{start[0]: 5} | {_file_contents[start[0] - 1]}')
            LOG.log(logging.INFO, '      | '  + (' ' * (start[1] - 1)) + ('^' * num_markers))

def emit_debug(message: str, *args, **kwargs) -> None:
    log(logging.DEBUG, message, *args, **kwargs)

def emit_info(message: str, *args, **kwargs) -> None:
    log(logging.INFO, message, *args, **kwargs)

def emit_warning(message: str, *args, **kwargs) -> None:
    log(logging.WARNING, message, *args, **kwargs)

def emit_error(message: str, *args, **kwargs) -> None:
    log(logging.ERROR, message, *args, **kwargs)

def emit_fatal(message: str, *args, **kwargs) -> None:
    log(logging.FATAL, message, *args, **kwargs)

class LogException(Exception):
    pass

class LogError(LogException):
    pass

class LogFatal(LogException):
    pass
