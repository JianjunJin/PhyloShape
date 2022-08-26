#!/usr/bin/env python

# from loguru import logger
# import time
# from tqdm import tqdm
# import io
#
#
# class TqdmToLogger(io.StringIO):
#     """
#         Output stream for TQDM which will output to logger module instead of
#         the StdOut.
#     """
#     def __init__(self, logger, level=None):
#         super(TqdmToLogger, self).__init__()
#         self.logger = logger
#         self.level = level or logger.getLevel
#         self.buf = ""
#
#     def write(self, buf):
#         self.buf = buf.strip('\r\n\t ')
#
#     def flush(self):
#         self.logger.log(self.level, self.buf)
#
#
# if __name__ == "__main__":
#     logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#
#     tqdm_out = TqdmToLogger(logger, level=logging.INFO)
#     for x in tqdm(range(100), file=tqdm_out, mininterval=30,):
#         time.sleep(.5)


from ipywidgets import IntProgress
from IPython.display import display


class ProgressLogger:
    def __init__(self, max_count: int = 100):
        self.max_count = max_count
        self.__logger = IntProgress(min=0, max=max_count)
        self.__counter = 0
        display(self.__logger)

    def update(self):
        if self.__counter <= self.max_count:
            self.__logger.value += 1
            self.__counter += 1

    def reset(self):
        self.__logger.value = 0
        self.__counter = 0


import sys


class ProgressText:
    def __init__(self, max_count: int = 100):
        self.max_count = max_count
        self.__counter = 0
        self.__percent = 0

    def update(self):
        self.__counter += 1
        if int(self.__counter/float(self.max_count) * 50) > self.__percent:
            sys.stdout.write("*")
            sys.stdout.flush()
            self.__percent += 1
        if self.__counter == self.max_count:
            sys.stdout.write("*" * (50 - self.__percent) + "\n")


