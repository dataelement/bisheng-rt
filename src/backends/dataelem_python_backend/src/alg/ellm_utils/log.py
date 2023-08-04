# flake8: noqa

import contextlib
import functools
import logging
import threading
import time

import colorlog

loggers = {}

log_config = {
    'DEBUG': {
        'level': 10,
        'color': 'purple'
    },
    'INFO': {
        'level': 20,
        'color': 'green'
    },
    'TRAIN': {
        'level': 21,
        'color': 'cyan'
    },
    'EVAL': {
        'level': 22,
        'color': 'blue'
    },
    'WARNING': {
        'level': 30,
        'color': 'yellow'
    },
    'ERROR': {
        'level': 40,
        'color': 'red'
    },
    'CRITICAL': {
        'level': 50,
        'color': 'bold_red'
    },
}


class Logger(object):
    def __init__(self, name: str = None):
        name = 'DATAELEM' if not name else name
        self.logger = logging.getLogger(name)

        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(self.__call__,
                                                   conf['level'])
            self.__dict__[key.lower()] = functools.partial(
                self.__call__, conf['level'])

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s',
            log_colors={
                key: conf['color']
                for key, conf in log_config.items()
            },
        )

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = 'DEBUG'
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._is_enable = True

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    def set_level(self, log_level: str):
        assert log_level in log_config, f'Invalid log level. Choose among {log_config.keys()}'
        self.logger.setLevel(log_level)

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: str, msg: str):
        if not self.is_enable:
            return

        self.logger.log(log_level, msg)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        """
        Continuously print a progress bar with rotating special effects.

        Args:
            msg(str): Message to be printed.
            interval(float): Rotation interval. Default to 0.1.
        """
        end = False

        def _printer():
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.info('{}: {}'.format(msg, flag))
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True


logger = Logger()
