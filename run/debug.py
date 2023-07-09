# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Debugger for Ideas."""

import spacy

from ..config import Config


class Debugger:
    """Debugger."""

    cfg = Config()

    def run_test(self) -> None:
        """Run Test."""
        # nlp = English()
        name = "en_core_web_trf"
        # name = "en_core_web_lg"
        nlp = spacy.load(name)


if __name__ == "__main__":
     Debugger().run_test()