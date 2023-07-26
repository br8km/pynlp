# -*- coding: utf-8 -*-

"""Find Spacy Semantic Patterns For Subreddit Content."""

import json
import string
from dataclasses import dataclass, asdict

import markdown
from bs4 import BeautifulSoup
from cleantext import clean
from emoji import demojize

import spacy
from spacy.language import Language
# from spacy.matcher import Matcher

from ..base.io import IO
from ..base.timer import timeit 
from ..config import Config
from ..core.emotion import EmotionDetectorRoberta
from ..core.sentiment import SentimentAnalysis


@dataclass
class Sentence:
    """Sentence."""

    text: str  # text content

    lem: list[str]  # list of token.lemma_
    ent: dict[str]  # dict of ent.text: ent.label_
    pos: list[str]  # list of token.pos_
    tag: list[str]  # list of token.tag_
    dep: list[str]  # list of token.dep_

    emotion: str
    sentiment: str

    logic: bool  # has logic words
    topic: bool  # has topic words


@dataclass
class Content:
    """Content."""

    cid: str  # Content ID string for submission|comment|reply
    type: str  # submission, comment or reply
    parent: str  # Parent content ID string

    sents: list[Sentence]


class SemanticPatternFinder:
    """Semantic Pattern Finder."""

    # filter by logic words
    # analysis semantic meanings
    # analysis emotion scores

    debug = True
    config = Config()

    logic_words = ["why", "what", "who", "where", "when", "which", "how"]
    topic_words = ["best", "top", "better", "worse", "worst", "bad", "most", "fact", "secret", "amazing", "guide", "review", "tutorial", "method"]

    def __init__(self) -> None:
        """Init."""
        self.load_models()

    @timeit
    def load_models(self) -> None:
        """Load Language Models."""
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("merge_entities")
        self.nlp.add_pipe("merge_noun_chunks")

        self.app_ed = EmotionDetectorRoberta()
        self.app_sa = SentimentAnalysis()

    def clean_up(self, text: str) -> str:
        """Clean up subreddit contents."""
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, features='html.parser')
        text = soup.get_text()
        text = demojize(text, use_aliases=True)
        return clean(text, lower=False)

    def is_rubbish(self, sentence: str) -> bool:
        """Check if text contain too much none-letter characters."""
        num = len(sentence)
        if num <= 10 or num >= 512:
            return True
        if num >= 60:
            letters = [char for char in sentence if char in string.ascii_letters]
            if len(letters) / num <= 0.5:
                return True
        return False

    def is_logic(self, lemmas: list[str]) -> bool:
        """Is list of token.lemma_ has logic words."""
        lemmas = [x.lower() for x in lemmas]
        for lemma in lemmas:
            if lemma in self.logic_words:
                return True
        return False

    def is_topic(self, lemmas: list[str]) -> bool:
        """Is list of token.lemma_ has topic words."""
        lemmas = [x.lower() for x in lemmas]
        for lemma in lemmas:
            if lemma in self.topic_words:
                return True
        return False

    def get_emotion(self, text: str) -> str:
        """Get Emotion from text string."""
        emotion = self.app_ed.get(text=text)
        # return emotion.tag
        return emotion

    def get_sentiment(self, text: str) -> str:
        """Get Sentiment from text string."""
        sentiment = self.app_sa.get(sentence=text)
        # return sentiment.mark
        return sentiment


    @timeit
    def load_contents(self, sr_name: str) -> list[dict]:
        """Load subreddit contents seperate by submission."""
        file = self.config.dir_tmp / f"sr_yt_scr-{sr_name}.json"
        assert file.is_file()
        return IO.load_list_dict(file)

    @timeit
    def load_spacy(self) -> Language:
        """Load spacy model with custom pipes."""
        nlp = spacy.load("en_core_web_sm")
        # print(nlp.pipe_names)
        # assert "transformer" not in nlp.pipe_names

        # nlp_coref = spacy.load("en_coreference_web_trf")
        # print(nlp_coref.pipe_names)

        # nlp.add_pipe("transformer", source=nlp_coref)
        # nlp.add_pipe("coref", source=nlp_coref)
        # nlp.add_pipe("span_resolver", source=nlp_coref)
        # nlp.add_pipe("span_cleaner", source=nlp_coref)

        return nlp

    def save(self, contents: list[Content]) -> bool:
        """Save analysis data."""
        file = self.config.dir_tmp / "pats.json"
        data = [asdict(x) for x in contents]
        IO.save_list_dict(file_name=file, file_data=data)
        return file.is_file()

    def process(self, texts: list[str]) -> list[Sentence]:
        """Process text string into list of Sentence."""
        result: list[Sentence] = []
        for text in texts:
            doc = self.nlp(text)
            for sent in doc.sents:
                if self.is_rubbish(sent.text):
                    continue
                sentence = Sentence(
                    text=sent.text, 
                    lem=[tk.lemma_ for tk in sent],
                    ent={ent.text: ent.label_ for ent in sent.ents},
                    pos=[tk.pos_ for tk in sent],
                    tag=[tk.tag_ for tk in sent],
                    dep=[tk.dep_ for tk in sent],
                    emotion=self.get_emotion(sent.text),
                    sentiment=self.get_sentiment(sent.text),
                    logic=self.is_logic(lemmas=[tk.lemma_ for tk in sent]),
                    topic=self.is_topic(lemmas=[tk.lemma_ for tk in sent]),
                )
                result.append(sentence)
        return result

    def analysis(self, data: list[dict]) -> list[Content]:
        """Analysis each piece of contents by spacy."""
        contents: list[Content] = []
        for sub in data:
            texts: list[str] = [sub["title"], sub["selftext"]]
            contents.append(
                Content(
                    cid=sub["id"],
                    type="submission",
                    parent="",
                    sents=self.process(texts=texts)
                )
            )

            for com in sub["comments"]:
                texts = [com["body"]]
                contents.append(
                    Content(
                        cid=com["id"],
                        type="comment",
                        parent=com["parent_id"],
                        sents=self.process(texts=texts)
                    )
                )

                for rep in com["replies"]:
                    texts = [rep["body"]]
                    contents.append(
                        Content(
                            cid=rep["id"],
                            type="reply",
                            parent=rep["parent_id"],
                            sents=self.process(texts=texts)
                        )
                    )

            if contents:
                self.save(contents=contents)

            if self.debug:
                break

        return contents

    @timeit
    def run(self) -> None:
        """Run."""
        self.debug = False
        sr_name = "nosurf"
        data = self.load_contents(sr_name=sr_name)
        contents = self.analysis(data=data)
        assert self.save(contents=contents)
 
    @timeit
    def run_test(self) -> None:
        """Run test."""
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("merge_entities")
        nlp.add_pipe("merge_noun_chunks")
        # nlp.add_pipe("sentencizer")

        text = "I saw The Who perform. Who did you see? John is a good man."
        text += "John locke is waiting for Tom Hakwin."
        text = "I saw the who perform."
        text = "John Locke is waiting for Tom Hakwin."
        text = "Are there any apps for iphone where i can track where i found a mushroom and what species i have found?"
        doc = nlp(text)
        print([(ent.label_, ent.text) for ent in doc.ents])
        print()
        data = {ent.text: ent.label_ for ent in doc.ents}
        print(json.dumps(data, indent=2))

    def to_do(self) -> None:
        """TO DO."""
        # product feature extraction


if __name__ == "__main__":
    SemanticPatternFinder().run_test()