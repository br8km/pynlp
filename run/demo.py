#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Demo."""

import json
from pathlib import Path
from datetime import datetime

import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

from ..core.emotion import EmotionDetectorT5
from ..core.emotion import EmotionDetectorRoberta
from ..core.sentiment import SentimentAnalysis, AspectBasedSentimentAnalysis
from ..core.ngrams import NgramComponent
from ..core.abbr import AbbreviationDetector
from ..core.hyponym import HyponymDetector


class AppDemo:
    """App Demo."""


    def print_line(self, num: int = 2) -> None:
        """Print linebreak."""
        print("\n" * num)

    def run_sentiment(self) -> None:
        """Run."""
        # nlp = SentimentAnalysis(use_aspect=True)
        # document = "Hello, world! Capitalism produces ecological crisis for the same reason it produces inequality: because the fundamental mechanism of capitalist growth is that capital must extract (from nature and labour) more than it gives in return."

        # sent = nlp.get_sentiment(document=document)
        # print(sent)
        # print(sent.positive, sent.negative, sent.neutral)
        # self.print_line()

        # sentences = [
        #     'The food we had yesterday was delicious',
        #     'My time in Italy was very enjoyable',
        #     'I found the meal to be tasty',
        #     'The internet was slow.',
        #     'Our experience was suboptimal'
        # ]
        # # aspects = ["food", "time"]
        # sents = nlp.get_aspect_sentiments(sentences)

        # for sent in sents:
        #     print(sent.name, sent.mark, sent)


        # self.print_line()
        # train_data = [
        #     ('Slow internet.', 'negative'),
        #     ('Delicious food', 'positive'),
        #     ('Suboptimal experience', 'negative'),
        #     ('Very enjoyable time', 'positive'),
        #     ('delicious food.', 'neg')
        # ]
        # nlp.train_aspect_sentiments(data=train_data)

        nlp = AspectBasedSentimentAnalysis()

        examples=[["1.) Instead of being at the back of the oven, the cord is attached at the front right side.","cord"],    ["The pan I received was not in the same league as my old pan, new is cheap feeling and does not have a plate on the bottom.","pan"],    ["The pan I received was not in the same league as my old pan, new is cheap feeling and does not have a plate on the bottom.","bottom"],    ["They seem much more durable and less prone to staining, retaining their white properties for a much longer period of time.","durability"],    ["It took some time to clean and maintain, but totally worth it!","clean"],    ["this means that not only will the smallest burner heat up the pan, but it will also vertically heat up 1\" of the handle.","handle"]]       
        for sentence, aspect in examples:
            start_time = datetime.now()
            print("\n---")
            result = nlp.get(sentence, aspect)
            print(str(result))
            print("sentiment.text: " + sentence)
            print("sentiment.aspect: " + aspect)
            print("sentiment.mark: " + result.mark)
            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))

    def run_emotion(self) -> None:
        """Run Emotion."""
        # assert nlp.prepare_for_training()
        # nlp = EmotionDetector(model_name="")
        # nlp.train(max_epochs=200, resume=True)

        # hug_model = "mrm8488/t5-base-finetuned-emotion"
        # hug_model = "SamLowe/roberta-base-go_emotions"

        # nlp = EmotionDetectorT5()
        nlp = EmotionDetectorRoberta()

        text = "i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute"

        start_time = datetime.now()
        emo = nlp.get(text) # Output: 'joy'
        end_time = datetime.now()
        print("text: " + text)
        print(type(emo), emo)
        print('Duration: {}'.format(end_time - start_time))
 
        text = "i have a feeling i kinda lost my best friend"

        start_time = datetime.now()
        emo = nlp.get(text) # Output: 'sadness'
        end_time = datetime.now()
        print("text: " + text)
        print(type(emo), emo)
        print('Duration: {}'.format(end_time - start_time))

    def run_grams(self) -> None:
        """Run n-grams."""
        nlp = spacy.load('en_core_web_md')  # or whatever model you downloaded
        nlp.add_pipe('spacy-ngram')  # default to document-level ngrams, removing stopwords

        text = 'Quark soup is an interacting localized assembly of quarks and gluons.'
        doc = nlp(text)

        print(doc._.ngram_1)
        # ['quark', 'soup', 'interact', 'localize', 'assembly', 'quark', 'gluon']

        print(doc._.ngram_2)
        # ['quark_soup', 'soup_interact', 'interact_localize', 'localize_assembly', 'assembly_quark', 'quark_gluon']

    def run_abbr(self) -> None:
        """Run Abbreviation."""
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

    def run_hyponym(self) -> None:
        """Run Hyponym Detector."""
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})

        doc = nlp("Keystone plant species such as fig trees are good for the soil.")

        print(doc._.hearst_patterns)

    def run_coref(self) -> None:
        """Run Coref."""
        nlp = spacy.load("en_core_web_sm")
        # ! Must Spacy.Version < 3.5, should train own model
        # nlp = spacy.load("en_coreference_web_trf")
        nlp.add_pipe("experimental_coref")
        doc = nlp("The cats were startled by the dog as it growled at them.") 
        print(doc.spans)

    def run_matcher(self) -> None:
        """Run Matcher."""
        nlp = spacy.load("en_core_web_sm")
        text = "This is a spaCy test."
        doc = nlp(text)
        for token in doc :
            print(token.text, token.pos_, token.lemma_)
        matcher = Matcher(nlp.vocab)
        phrase_matcher = PhraseMatcher(nlp.vocab)
        lexicon = ["like", "love", "I like"]
        lexicon_2 = ["do not like", "enjoy"]
        patterns = [
            # A pronoun + "love"
            [{"POS": "PRON"}, {"LEMMA": "love"}],
            # A pronoun + the  verb "like"
            [{"POS": "PRON"}, {"LEMMA": "like", "POS": "VERB"}],
        ]
        patterns_2 = [
            # a pronoun + "love" with 1 or 0 word between them (ex : "I really love")
            [{"POS": "PRON"}, {"IS_ALPHA": True, "OP": "?"}, {"LEMMA": "love"}]
        ]
        # Add the lexicons to the Phrase matcher
        phrase_matcher.add("some_lemmas", [nlp(word) for word in lexicon])
        phrase_matcher.add("more_lemmas", [nlp(word) for word in lexicon_2])
        # Add the patterns to Matcher
        matcher.add("some_patterns", patterns)
        matcher.add("1_more_pattern", patterns_2)

        doc = nlp("Do you like it or do you love it ?")
        # Apply your Matchers to the doc
        matches = phrase_matcher(doc)
        matches += matcher(doc)
        # Print the Hash value, start index and end index of the matches
        for result in matches:
            print(result)# Print the number of matches
        print("Total matches found:", len(matches))# Print results
        for match_id, start, end in matches:
                # Prints the rule id
                print(nlp.vocab.strings[match_id]) 
            # Prints the text matched
                print(doc[start:end])

    def run(self) -> None:
        """Run."""
        self.run_sentiment()
        # self.run_emotion()
        # self.run_grams()
        # self.run_abbr()
        # self.run_hyponym()
        # self.run_coref()
        # self.run_matcher()


if __name__ == "__main__":
    AppDemo().run()