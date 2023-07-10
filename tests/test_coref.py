#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Coreference Resolution."""

from datetime import datetime

from ..base.timer import timeit

import spacy


class TestCoref:
    """Test Coreference Resolution."""

    @timeit
    def run_exp(self) -> None:
        """Test Experimental."""
        start = datetime.now()
        nlp = spacy.load("en_core_web_sm")
        end = datetime.now()
        print(f"nlp.load.sm: {end - start} seconds.")

        assert "transformer" not in nlp.pipe_names

        start = datetime.now()
        nlp_coref = spacy.load("en_coreference_web_trf")
        end = datetime.now()
        print(f"nlp.load.trf: {end - start} seconds.")

        nlp.add_pipe("transformer", source=nlp_coref)
        nlp.add_pipe("coref", source=nlp_coref)
        nlp.add_pipe("span_resolver", source=nlp_coref)
        nlp.add_pipe("span_cleaner", source=nlp_coref)

        start = datetime.now()
        # nlp.add_pipe("experimental_coref")
        txt = "The cats were startled by the dog as it growled at them."
        print(txt)
        doc = nlp(txt)
        print(doc.spans)
        end = datetime.now()
        print(f"nlp.get.coref: {end - start} seconds.")

    @timeit
    def run(self) -> None:
        """Run."""
        nlp = spacy.load("en_coreference_web_trf")
        txt = "The cats were startled by the dog as it growled at them."
        print(txt)
        doc = nlp(txt)
        print(doc.spans)


if __name__ == "__main__":
    TestCoref().run_exp()