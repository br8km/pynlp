#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Text n-grams Generator."""

# Reference: https://github.com/kpwhri/spacy-ngram#usage

from __future__ import annotations
from collections import deque, defaultdict

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span


@Language.factory(
    'spacy-ngram',
    default_config={
        'extension_name': 'ngram',
        'ngrams': (1, 2),
        'include_bos': False,
        'include_eos': False,
        'sentence_level': False,
        'doc_level': True,
    },
)
def create_ngram_component(
                            nlp: Language,
                            name: str,
                            extension_name: str,
                            ngrams: tuple[int, ...],
                            include_bos: bool,
                            include_eos: bool,
                            sentence_level: bool,
                            doc_level: bool) -> NgramComponent:
    if not sentence_level and not doc_level:
        raise ValueError(
            'Ngram target must be specified at sentence or document-level in the config: `sentence_level=True`')

    if isinstance(ngrams, int):
        ngrams = (ngrams,)

    if ngrams is None or len(ngrams) == 0:
        raise ValueError(
            'No ngram levels specified: try updating config to include unigrams: `ngrams=(1,)'
        )

    return NgramComponent(nlp, extension_name, ngrams=ngrams, include_bos=include_bos, include_eos=include_eos, sentence_level=sentence_level, doc_level=doc_level)


class NgramComponent:
    """Spacy pipeline for ngram extraction."""

    def __init__(self,
                 nlp: Language,
                 extension_name: str,
                 include_bos=False,
                 include_eos=False,
                 sentence_level=False,
                 doc_level=True,
                 ngrams=None) -> None:

        if isinstance(ngrams, int):
            ngrams = (ngrams,)

        for count in ngrams:
            ext = f'{extension_name}_{count}'
            if doc_level and not Doc.has_extension(ext):
                Doc.set_extension(ext, default=False, force=True)
            if sentence_level and not Span.has_extension(ext):
                Span.set_extension(ext, default=False, force=True)

        self.nlp = nlp
        self.extension_name = extension_name
        self.include_bos = include_bos
        self.include_eos = include_eos
        self.sentence_level = sentence_level
        self.doc_level = doc_level
        self.ngrams = ngrams

    def add_sentence_ngrams(self, doc: Doc):
        """Add sentence-level ngrams"""
        for sent in doc.sents:
            self.get_ngrams(sent)

    def add_document_ngrams(self, doc: Doc):
        """Add document-level ngrams"""
        self.get_ngrams(doc)

    def get_ngrams(self, sequence: Doc | Span):
        """Get ngrams from Doc or Span by iterating through the words"""
        curr_words = deque(maxlen=max(self.ngrams))  # TODO: backfill with BOS?
        curr_ngrams = defaultdict(list)
        for word in self.itertokens(sequence):
            curr_words.append(word)
            for count, ngram in self.ngramize(curr_words):
                curr_ngrams[count].append(ngram)
        for count in self.ngrams:
            sequence._.set(f'{self.extension_name}_{count}', curr_ngrams[count])

    def ngramize(self, sequence: deque):
        """Create relevant ngrams for a given sequence"""
        for count in self.ngrams:
            if len(sequence) >= count:
                yield count, '_'.join(sequence[i] for i in range(-count, 0))

    def itertokens(self, sequence: Doc | Span):
        """Return next relevant lemma from the sequence"""
        if self.include_bos:
            yield '<BOS>'
        for token in sequence:
            if token.is_stop or token.is_punct or token.is_digit:
                continue
            lemma = token.lemma_.lower()
            if lemma in self.nlp.Defaults.stop_words:
                continue
            yield lemma
        if self.include_eos:
            yield '<EOS>'

    def __call__(self, doc: Doc):
        """Pipeline entrypoint"""
        if self.sentence_level:
            self.add_sentence_ngrams(doc)
        if self.doc_level:
            self.add_document_ngrams(doc)
        return doc


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
        sentence = list(doc.sents)

        print(sentence._.ngram_1)
        # raises AttributeError
        print(sentence._.ngram_2)  # returns list of bigrams
        print(sentence._.ngram_3)  # returns list of trigrams