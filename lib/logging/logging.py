import time
import sys
import os


class Logger(object):
    def __init__(self):
        self.log_file = None

    def set_log_file(self, path):
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()
        self.log_file = open(path, 'w', encoding="utf-8", buffering=1)

    def log_print(self, text, to_console=True):
        if self.log_file is not None:
            if not isinstance(text, str):
                text = str(text)
            self.log_file.write(text + "\n")
            self.log_file.flush()
            if to_console:
                print(text, file=sys.stderr)

    def close_file(self):
        self.log_file.close()
