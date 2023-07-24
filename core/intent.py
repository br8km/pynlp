#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Intent Detector."""

from typing import Any

from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc


product_name = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
product_category = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}  # tools, apps, websites, sites, solutions, software, etc.
product_candidate = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
product_feature = {}
customer_object = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
compititer_name = {}
customer_pain = {}

punct = {"IS_PUNCT": True, "OP": "?"}
det = {"ORTH": "*", "OP": "*"}



# I wish there was
# 

INTENT_PATTERNS: list[dict[str, Any]] = [
    {
        "scope": "customer",
        "kind": "customer_ideas",
        "label": "alternative_for",
        "pattern": [
            {"LEMMA": "alternative"},
            {"LEMMA": "for"},
            product,
        ],
    },
    {
        "scope": "customer",
        "kind": "customer_ideas",
        "label": "any_for",
        "pattern": [
            {"LEMMA": "any"},
            product_category,
            {"LEMMA": "for"},
            target,
        ],
    },
    {
        "scope": "customer",
        "kind": "customer_ideas",
        "label": "any_like",
        "pattern": [
            {"LEMMA": "any"},
            product_category,
            {"LEMMA": "like"},
            product,
        ],
    }

]


@Language.factory("intent_detector")
class IntentDetector:
    """
    A spaCy pipe for intent detect.
    This class sets the following attributes:

    - `Doc._.intents`: A List[Tuple[str, Span, Span]] corresonding to
       the matching predicate, extracted general term and specific term
       that matched a grammar pattern.

    The pipe can be used with an instantiated spacy model like so:
    ```
    # add the intent_detector
    nlp.add_pipe('intent_detector')

    Parameters
    ----------

    nlp: `Language`, a required argument for spacy to use this as a factory
    name: `str`, a required argument for spacy to use this as a factory
    category: `str`, which category of intent patterns to look for
    """

    def __init__(self, 
                 nlp: Language,
                 name: str = "intent_detector",
                 scope: str = "customer"):
        self.nlp = nlp
        self.name = name
        self.scope = scope

        self.patterns = [p for p in INTENT_PATTERNS if p["scope"]==scope]
        self.matcher = Matcher(self.nlp.vocab)

        Doc.set_extension("intent_patterns", default=[], force=True)

        # add patterns to matcher
        for pattern in self.patterns:
            self.matcher.add(pattern["label"], [pattern["pattern"]])

    def process(self, token: Token, doc: Doc) -> None:
        """Process Token example."""
        raise NotImplementedError

    def __call__(self, doc: Doc):
        """Find matches in doc."""
        matches = self.matcher(doc)

        # If none are found then return None
        if not matches:
            return doc

        return doc