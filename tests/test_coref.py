#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Coreference Resolution."""

from ..base.timer import timeit

import spacy


class TestCoref:
    """Test Coreference Resolution."""

    @timeit
    def run(self) -> None:
        """Run."""
        nlp = spacy.load("en_coreference_web_trf")
        txt = "The cats were startled by the dog as it growled at them."
        print(txt)
        doc = nlp(txt)
        print(doc.spans)


if __name__ == "__main__":
    TestCoref().run()