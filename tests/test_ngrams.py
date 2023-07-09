#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test n-Grams Generation."""

# Reference: https://github.com/kpwhri/spacy-ngram#usage

import spacy

# from spacy_ngram import NgramComponent
from ..core.ngrams import create_ngram_component, NgramComponent


class TestNGrams:
    """Test n-grams."""

    def example_one(self) -> None:
        """Example One."""
        nlp = spacy.load('en_core_web_sm')  # or whatever model you downloaded
        nlp.add_pipe('spacy-ngram')  # default to document-level ngrams, removing stopwords

        text = 'Quark soup is an interacting localized assembly of quarks and gluons.'
        doc = nlp(text)

        print(doc._.ngram_1)
        # ['quark', 'soup', 'interact', 'localize', 'assembly', 'quark', 'gluon']

        print(doc._.ngram_2)
        # ['quark_soup', 'soup_interact', 'interact_localize', 'localize_assembly', 'assembly_quark', 'quark_gluon']

    def example_two(self) -> None:
        """Example Two."""
        nlp = spacy.load('en_core_web_sm')  # or whatever model you downloaded
        nlp.add_pipe('spacy-ngram', config={
            'sentence_level': True,  # initialize sentence-level ngrams
            'doc_level': False,  # skip processing at document-level
            'ngrams': (2, 3),  # bi- and trigram only
        })
        text = 'Quark soup is an interacting localized assembly of quarks and gluons.'
        doc = nlp(text)
        sentence = list(doc.sents)[0]

        try:
            print(sentence._.ngram_1)
            # raises AttributeError
        except AttributeError:
            print("raise AttributeError")

        print(sentence._.ngram_2)  # returns list of bigrams
        print(sentence._.ngram_3)  # returns list of trigrams

    def run(self) -> None:
        """Run."""
        self.example_one()
        self.example_two()

    
if __name__ == "__main__":
    TestNGrams().run()