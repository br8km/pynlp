#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Text Sentiment Analysis."""

# Reference: 
# - https://github.com/SamEdwardes/spaCyTextBlob
# - https://towardsdatascience.com/aspect-based-sentiment-analysis-using-spacy-textblob-4c8de3e0d2b9
# - https://github.com/Vishnunkumar/eng_spacysentiment

# - [bad] https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis

from dataclasses import dataclass

import spacy
from spacy.language import Language
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn


@dataclass
class Sentiment:
    """Sentiment."""

    polarity: float  # [-1.0, 1.0]
    subjectivity: float  # [0.0, 1.0]

    @property
    def positive(self) -> bool:
        """Is Positive or Not."""
        return bool(self.subjectivity > 0.5 and self.polarity > 0)

    @property
    def negative(self) -> bool:
        """Is Negative or Not."""
        return bool(self.subjectivity > 0.5 and self.polarity < 0)

    @property
    def neutral(self) -> bool:
        """Is Neutral or Not."""
        return bool(not self.positive and not self.negative)

    @property
    def mark(self) -> str:
        """Sentment Mark."""
        return "positive" if self.positive else "negative" if self.negative else "neutral"


@dataclass
class AspectSentiment:
    """Aspect Sentiment."""

    aspect: str

    positive: float
    negative: float
    neutral: float

    @property
    def mark(self) -> str:
        """Sentiment Mark."""
        keys = ("positive", "negative", "neutral")
        _mark, _score = "", 0
        for key in keys:
            value = getattr(self, key, 0)
            if value > _score:
                _mark = key
                _score = value
        return _mark

        


class SentimentAnalysis:
    """Sentiment Analysis."""

    nlp: Language

    def __init__(self,
                 use_aspect: bool = False,
                 model_name: str = "en_core_web_md") -> None:
        """Init Sentiment Analysis."""
        self.use_aspect = use_aspect
        self.model_name = model_name
        if self.use_aspect and self.model_name:
            self.nlp = spacy.load(model_name)

    def get_sentiment(self, document: str) -> Sentiment:
        """Get Sentiment for document string."""
        blob = TextBlob(document)
        # default sentiment type not assessible, use custom one here.
        return Sentiment(
            polarity=blob.polarity,
            subjectivity=blob.subjectivity
        )

    def _train_aspect_sentiments(self, data: list[tuple[str, str]]) -> None:
        """Train custom aspect based sentiment analyzer."""
        cl = NaiveBayesClassifier(data)
        blob = TextBlob("Delicious food. Very Slow internet. Suboptimal experience. Enjoyable food.", classifier=cl)
        for s in blob.sentences:
            print(s, s.classify())

    def _get_aspect_sentiments(self, sentences: list[str]) -> list[AspectSentiment]:
        """Get Aspect Sentiment for list of sentence string."""
        # aspects = list(set(w.lower() for w in aspects))

        # results: list[AspectSentiment] = [
        #     AspectSentiment(name=w, polarity=0.0, subjectivity=0.0)
        #     for w in aspects
        # ]

        results: list[AspectSentiment] = []

        data: list[dict] = []

        for sentence in sentences:
            doc = self.nlp(sentence)
            descriptive_term = ""
            target = ''
            for token in doc:
                if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
                    target = token.text
                if token.pos_ == "ADJ":
                    prepend = ""
                    for child in token.children:
                        if child.pos_ != 'ADV':
                            continue
                        prepend += child.text + ' '
                    descriptive_term = prepend + token.text
            data.append({'aspect': target, 'description': descriptive_term})

        for item in data:
            item['sentiment'] = TextBlob(item['description']).sentiment
            results.append(
                AspectSentiment(
                    name=item["aspect"],
                    polarity=item["sentiment"].polarity,
                    subjectivity=item["sentiment"].subjectivity,
                )
            )

        return results


class AspectBasedSentimentAnalysis:
    """Aspect Based Sentiment Analysis."""

    def __init__(self) -> None:
        """Init."""
        self.model_name = "yangheng/deberta-v3-base-absa-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def get(self, sentence: str, aspect: str) -> AspectSentiment:
        """Get Sentiment Score from text str and list of aspect string."""
        keys = ["negative","neutral","positive"]
        input_str = "[CLS]" + sentence + "[SEP]" + aspect + "[SEP]"
        # input_str = "[CLS] when tables opened up, the manager sat another party before us. [SEP] manager [SEP]"
        inputs = self.tokenizer(input_str, return_tensors="pt")
        outputs = self.model(**inputs)
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs.logits)
        results = [round(i,4) for i in outputs.tolist()[0]]
        # print(result)
        data = dict(zip(keys, results))
        return AspectSentiment(
            aspect=aspect,
            positive=data["positive"],
            negative=data["negative"],
            neutral=data["neutral"],
        )
