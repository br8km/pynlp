# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Debugger for Ideas."""

from pathlib import Path

from ..base.timer import timeit
from ..config import Config


class Debugger:
    """Debugger."""

    config = Config()

    dir_app = Path(__file__).parent

    def __init__(self) -> None:
        """Init."""
        self.name = "name"
        for key in dir(self):
            if key.startswith("__"):
                continue
            print(key, getattr(self, key, None))

    @timeit
    def get(self) -> None:
        """Get."""
        pass



if __name__ == "__main__":
     Debugger().get()