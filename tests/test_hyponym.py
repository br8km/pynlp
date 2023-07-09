#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Hyponym Detection."""

import spacy


from ..core.hyponym import HyponymDetector


class TestHyponym:
    """Test Hyponym Detector."""

    def run(self) -> None:
        """Run Example."""
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})

        doc = nlp("Keystone plant species such as fig trees are good for the soil.")

        print(doc._.hearst_patterns)


if __name__ == "__main__":
    TestHyponym().run()