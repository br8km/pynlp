#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Intent Patterns."""

from pathlib import Path

import spacy

from ..base.io import IO
from ..core.intent import Intent
from ..core.intent import IntentFinder


class TestIntentFinder:
    """Test Intent Finder."""

    app = IntentFinder()
    intents: list[Intent]

    def __init__(self, file_data: Path) -> None:
        """Init."""
        self.intents = IO.load_list_dict(file_nae=file_data)

    def load(self) -> list[Intent]:
        """Load list of Grammar Patterns."""
        raise NotImplementedError

    def save(self) -> bool:
        """Save Intents into local file."""
        raise NotImplementedError

    def update(self, intents: list[Intent]) -> bool:
        """Update list of Grammar patterns."""
        raise NotImplementedError

    def get(self, context: str, category: str, niche: str) -> list[Intent]:
        """Get Best Matched IntentPatterns."""
        return [x for x in self.intents if x.context==context and x.category==category and x.niche==niche]

    def run(self) -> None:
        """Run."""
        nlp = spacy.load("en_core_web_sm")
        config = {
            "context": "forum.subreddit",
            "category": "game",
            "niche":    "game hacks",
        }
        nlp.add_pipe("intent_finder", last=True, config=config)
        document = """Hello World."""
        doc = nlp(document)
        print(doc._.parsed_intents)


if __name__ == "__main__":
    TestIntentFinder().run()