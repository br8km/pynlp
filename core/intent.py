#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Intent Patterns."""

from typing import Any
from dataclasses import dataclass

from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc


@dataclass
class Intent:
    """Intent."""
    
    # kind: str  # Knowedge, Information, etc.
    context: str  # Article, Post, Story, Book, Movie, Song, etc.
    category: str  # Sports, Celebs, Movie, etc.
    niche: str  #  sport vechicle, video gaming etc.

    lable: str
    rules: list[dict[str, Any]]


@Language.factory("intent_finder")
class IntentFinder:
    """
    A spaCy pipe for intent detect.
    This class sets the following attributes:

    - `Doc._.intents`: A List[Tuple[str, Span, Span]] corresonding to
       the matching predicate, extracted general term and specific term
       that matched a grammar pattern.

    The pipe can be used with an instantiated spacy model like so:
    ```
    # add the intent_finder
    nlp.add_pipe('intent_finder', config={'extended': True}, last=True)
    nlp.add_pipe(")

    Parameters
    ----------

    nlp: `Language`, a required argument for spacy to use this as a factory
    name: `str`, a required argument for spacy to use this as a factory
    extended: `bool`, whether to use the extended Hearts patterns or not
    """

    def __init__(
        self, nlp: Language,
        context: str,
        category: str,
        niche: str,
        name: str = "smart_doctor",
    ):
        self.nlp = nlp
        self.name = name
        self.context = context
        self.category = category
        self.niche = niche

        self.patterns = []

        self.matcher = Matcher(self.nlp.vocab)

        Doc.set_extension("grammar_patterns", default=[], force=True)

        self.first = set()
        self.last = set()

        # add patterns to matcher
        for pattern in self.patterns:
            self.matcher.add(pattern["label"], [pattern["pattern"]])

            # gather list of predicates where the hypernym appears first
            if pattern["position"] == "first":
                self.first.add(pattern["label"])

            # gather list of predicates where the hypernym appears last
            if pattern["position"] == "last":
                self.last.add(pattern["label"])
    
    def process(self, token: Token) -> None:
        """Process Token example."""
        raise NotImplementedError

    def __call__(self, doc: Doc):
        """Find matches in doc."""
        matches = self.matcher(doc)

        # If none are found then return None
        if not matches:
            return doc

        return doc