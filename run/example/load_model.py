# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load Models with extra source model pipes."""

import spacy

from ..base.common import timeit


class Debugger:
    """Debugger."""

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