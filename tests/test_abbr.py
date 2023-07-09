#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Abbreviation Detector."""


import spacy


from ..core.abbr import AbbreviationDetector


class TestAbbr:
    """Test Abbreviation."""

    def example(self) -> None:
        """Example."""
        nlp = spacy.load("en_core_web_sm")

        # Add the abbreviation pipe to the spacy pipeline.
        nlp.add_pipe("abbreviation_detector")

        doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
                inherited motor neuron disease caused by the expansion \
                of a polyglutamine tract within the androgen receptor (AR). \
                SBMA can be caused by this easily.")

        print("Abbreviation", "\t", "Definition")
        for abrv in doc._.abbreviations:
            print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")


if __name__ == "__main__":
    TestAbbr().example()