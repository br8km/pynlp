#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Text Hyponym Detector."""

# https://github.com/allenai/scispacy/blob/main/scispacy/hyponym_detector.py


from typing import List, Dict, Any

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from spacy.language import Language


"""
BSD 3-Clause License

Copyright (c) 2020, Fourthought
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

hypernym = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
hyponym = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
punct = {"IS_PUNCT": True, "OP": "?"}
det = {"ORTH": "*", "OP": "*"}

BASE_PATTERNS: List[Dict[str, Any]] = [
    # '(NP_\\w+ (, )?such as (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "such_as",
        "pattern": [hypernym, punct, {"LEMMA": "such"}, {"LEMMA": "as"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "include",
        "pattern": [hypernym, punct, {"LEMMA": "include"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "especially",
        "pattern": [hypernym, punct, {"LEMMA": "especially"}, det, hyponym],
        "position": "first",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last'
    {
        "label": "other",
        "pattern": [
            hyponym,
            punct,
            {"LEMMA": {"IN": ["and", "or"]}},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
]

EXTENDED_PATTERNS = [
    # '(NP_\\w+ (, )?which may include (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "which_may_include",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "may"},
            {"LEMMA": "include"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?which be similar to (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "which_be_similar_to",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "be"},
            {"LEMMA": "similar"},
            {"LEMMA": "to"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?example of this be (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "example_of_this_be",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "example"},
            {"LEMMA": "of"},
            {"LEMMA": "this"},
            {"LEMMA": "be"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?type (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "type",
        "pattern": [hypernym, punct, {"LEMMA": "type"}, punct, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?mainly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "mainly",
        "pattern": [hypernym, punct, {"LEMMA": "mainly"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?mostly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "mostly",
        "pattern": [hypernym, punct, {"LEMMA": "mostly"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?notably (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "notably",
        "pattern": [hypernym, punct, {"LEMMA": "notably"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?particularly (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "particularly",
        "pattern": [hypernym, punct, {"LEMMA": "particularly"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?principally (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "principally",
        "pattern": [hypernym, punct, {"LEMMA": "principally"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?in particular (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "in_particular",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "in"},
            {"LEMMA": "particular"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?except (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "except",
        "pattern": [hypernym, punct, {"LEMMA": "except"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?other than (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "other_than",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": {"IN": ["other", "oth"]}},
            {"LEMMA": "than"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?e.g. (, )?(NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "eg",
        "pattern": [hypernym, punct, {"LEMMA": {"IN": ["e.g.", "eg"]}}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?i.e. (, )?(NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "ie",
        "pattern": [hypernym, punct, {"LEMMA": {"IN": ["i.e.", "ie"]}}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?for example (, )?(NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "for_example",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "for"},
            {"LEMMA": "example"},
            punct,
            det,
            hyponym,
        ],
        "position": "first",
    },
    # 'example of (NP_\\w+ (, )?be (NP_\\w+ ? '(, )?(and |or )?)+)', 'first'
    {
        "label": "example_of_be",
        "pattern": [
            {"LEMMA": "example"},
            {"LEMMA": "of"},
            hypernym,
            punct,
            {"LEMMA": "be"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?like (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "like",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "like"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # 'such (NP_\\w+ (, )?as (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "such_NOUN_as",
        "pattern": [
            {"LEMMA": "such"},
            hypernym,
            punct,
            {"LEMMA": "as"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?whether (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "whether",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "whether"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?compare to (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "compare_to",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "compare"},
            {"LEMMA": "to"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?among -PRON- (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "among_-PRON-",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "among"},
            {"LEMMA": "-PRON-"},
            det,
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )? (NP_\\w+ ? (, )?(and |or )?)+ for instance)', 'first'
    {
        "label": "for_instance",
        "pattern": [
            hypernym,
            punct,
            det,
            hyponym,
            {"LEMMA": "for"},
            {"LEMMA": "instance"},
        ],
        "position": "first",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?any other NP_\\w+)', 'last'
    {
        "label": "and-or_any_other",
        "pattern": [
            det,
            hyponym,
            punct,
            {"DEP": "cc"},
            {"LEMMA": "any"},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?some other NP_\\w+)', 'last'
    {
        "label": "some_other",
        "pattern": [
            det,
            hyponym,
            punct,
            {"DEP": "cc", "OP": "?"},
            {"LEMMA": "some"},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?be a NP_\\w+)', 'last'
    {
        "label": "be_a",
        "pattern": [
            det,
            hyponym,
            punct,
            {"LEMMA": "be"},
            {"LEMMA": "a"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "like_other",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?like other NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "like"},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "one_of_the",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?one of the NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "one"},
            {"LEMMA": "of"},
            {"LEMMA": "the"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "one_of_these",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?one of these NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "one"},
            {"LEMMA": "of"},
            {"LEMMA": "these"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "one_of_those",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?one of those NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"DEP": "cc", "OP": "?"},
            {"LEMMA": "one"},
            {"LEMMA": "of"},
            {"LEMMA": "those"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "be_example_of",
        "pattern": [
            # '((NP_\\w+ ?(, )?)+(and |or )?be example of NP_\\w+)',
            # added optional "an" to spaCy pattern for singular vs. plural
            # 'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "be"},
            {"LEMMA": "an", "OP": "?"},
            {"LEMMA": "example"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "which_be_call",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which be call NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "be"},
            {"LEMMA": "call"},
            hypernym,
        ],
        "position": "last",
    },
    #
    {
        "label": "which_be_name",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which be name NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "be"},
            {"LEMMA": "name"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "a_kind_of",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and|or)? a kind of NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "a"},
            {"LEMMA": "kind"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
    #                     '((NP_\\w+ ?(, )?)+(and|or)? kind of NP_\\w+)', - combined with above
    #                     'last'
    {
        "label": "form_of",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and|or)? form of NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "a", "OP": "?"},
            {"LEMMA": "form"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "which_look_like",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which look like NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "look"},
            {"LEMMA": "like"},
            hyponym,
        ],
        "position": "last",
    },
    {
        "label": "which_sound_like",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which sound like NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "sound"},
            {"LEMMA": "like"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "type",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )? NP_\\w+ type)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "type"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "compare_with",
        "pattern": [
            #                     '(compare (NP_\\w+ ?(, )?)+(and |or )?with NP_\\w+)',
            #                     'last'
            {"LEMMA": "compare"},
            det,
            hyponym,
            punct,
            {"LEMMA": "with"},
            hypernym,
        ],
        "position": "last",
    },
    #             {"label" : "as", "pattern" : [
    # #                     '((NP_\\w+ ?(, )?)+(and |or )?as NP_\\w+)',
    # #                     'last'
    #                 hyponym, punct, {"LEMMA" : "as"}, hypernym
    #             ], "position" : "last"},
    {
        "label": "sort_of",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and|or)? sort of NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "sort"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
]

@Language.factory("hyponym_detector")
class HyponymDetector:
    """
    A spaCy pipe for detecting hyponyms using Hearst patterns.
    This class sets the following attributes:

    - `Doc._.hearst_patterns`: A List[Tuple[str, Span, Span]] corresonding to
       the matching predicate, extracted general term and specific term
       that matched a Hearst pattern.

    Parts of the implementation taken from
    https://github.com/mmichelsonIF/hearst_patterns_python/blob/master/hearstPatterns/hearstPatterns.py
    and
    https://github.com/Fourthought/CNDPipeline/blob/master/cndlib/hpspacy.py

    The pipe can be used with an instantiated spacy model like so:
    ```
    # add the hyponym detector
    nlp.add_pipe('hyponym_detector', config={'extended': True}, last=True)

    Parameters
    ----------

    nlp: `Language`, a required argument for spacy to use this as a factory
    name: `str`, a required argument for spacy to use this as a factory
    extended: `bool`, whether to use the extended Hearts patterns or not
    """

    def __init__(
        self, nlp: Language, name: str = "hyponym_detector", extended: bool = False
    ):
        self.nlp = nlp

        self.patterns = BASE_PATTERNS
        if extended:
            self.patterns.extend(EXTENDED_PATTERNS)

        self.matcher = Matcher(self.nlp.vocab)

        Doc.set_extension("hearst_patterns", default=[], force=True)

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

    def expand_to_noun_compound(self, token: Token, doc: Doc):
        """
        Expand a token to it's noun phrase based
        on a simple POS tag heuristic.
        """

        start = token.i
        while True:
            if start - 1 < 0:
                break
            previous_token = doc[start - 1]
            if previous_token.pos_ in {"PROPN", "NOUN", "PRON"}:
                start -= 1
            else:
                break

        end = token.i + 1
        while True:
            if end >= len(doc):
                break
            next_token = doc[end]
            if next_token.pos_ in {"PROPN", "NOUN", "PRON"}:
                end += 1
            else:
                break

        return doc[start:end]

    def find_noun_compound_head(self, token: Token):
        while token.head.pos_ in {"PROPN", "NOUN", "PRON"} and token.dep_ == "compound":
            token = token.head
        return token

    def __call__(self, doc: Doc):
        """
        Runs the matcher on the Doc object and sets token and
        doc level attributes for hypernym and hyponym relations.
        """
        # Find matches in doc
        matches = self.matcher(doc)

        # If none are found then return None
        if not matches:
            return doc

        for match_id, start, end in matches:
            predicate = self.nlp.vocab.strings[match_id]

            # if the predicate is in the list where the hypernym is last, else hypernym is first
            if predicate in self.last:
                hypernym = doc[end - 1]
                hyponym = doc[start]
            else:
                # An inelegent way to deal with the "such_NOUN_as pattern"
                # since the first token is not the hypernym.
                if doc[start].lemma_ == "such":
                    start += 1
                hypernym = doc[start]
                hyponym = doc[end - 1]

            hypernym = self.find_noun_compound_head(hypernym)
            hyponym = self.find_noun_compound_head(hyponym)

            # For the document level, we expand to contain noun phrases.
            hypernym_extended = self.expand_to_noun_compound(hypernym, doc)
            hyponym_extended = self.expand_to_noun_compound(hyponym, doc)

            doc._.hearst_patterns.append(
                (predicate, hypernym_extended, hyponym_extended)
            )

            for token in hyponym.conjuncts:
                token_extended = self.expand_to_noun_compound(token, doc)
                if token != hypernym and token is not None:
                    doc._.hearst_patterns.append(
                        (predicate, hypernym_extended, token_extended)
                    )

        return doc


class TestHyponym:
    """Test Hyponym Detector."""

    def example(self) -> None:
        """Example."""
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})

        doc = nlp("Keystone plant species such as fig trees are good for the soil.")

        print(doc._.hearst_patterns)