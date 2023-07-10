# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Debugger for Ideas."""

import spacy

from ..base.common import timeit
from ..config import Config


class Debugger:
    """Debugger."""

    cfg = Config()

    @timeit
    def run_test(self) -> None:
        """Run Test."""
        nlp = spacy.load("en_core_web_sm")
        assert "transformer" not in nlp.pipe_names
        nlp_coref = spacy.load("en_coreference_web_trf")
        nlp.add_pipe("transformer", source=nlp_coref)
        nlp.add_pipe("coref", source=nlp_coref)
        nlp.add_pipe("span_resolver", source=nlp_coref)
        nlp.add_pipe("span_cleaner", source=nlp_coref)


if __name__ == "__main__":
     Debugger().run_test()