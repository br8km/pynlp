# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Debugger for Ideas."""

import spacy
from spacy import explain
# from spacytextblob.spacytextblob import SpacyTextBlob

from ..base.timer import timeit


class Debugger:
    """Debugger."""

    @timeit
    def run_test(self) -> None:
        """Run Test."""
        nlp = spacy.load("en_core_web_sm")
        # nlp.add_pipe('spacytextblob')
        # Merge noun phrases and entities for easier analysis
        nlp.add_pipe("merge_entities")
        nlp.add_pipe("merge_noun_chunks")

        TEXTS = [
            "Net income was $9.4 million compared to the prior year of $2.7 million.",
            "Revenue exceeded twelve billion dollars, with a loss of $1b.",
        ]
        for doc in nlp.pipe(TEXTS):
            for token in doc:
                if token.ent_type_ == "MONEY":
                    # We have an attribute and direct object, so check for subject
                    if token.dep_ in ("attr", "dobj"):
                        subj = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                        if subj:
                            print(subj[0], "-->", token)
                    # We have a prepositional object with a preposition
                    elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                        print(token.head.head, "-->", token)



if __name__ == "__main__":
     Debugger().run_test()