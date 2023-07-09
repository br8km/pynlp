#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Sentiment Analysis."""

from datetime import datetime

from ..core.sentiment import Sentiment, AspectSentiment, SentimentAnalysis, AspectBasedSentimentAnalysis


class TestSentimentAnalysis:
    """Test Sentiment Analysis."""

    def get_sentiment(self) -> None:
        """Get Sentiment."""
        nlp = SentimentAnalysis()
        document = "Hello, world! Capitalism produces ecological crisis for the same reason it produces inequality: because the fundamental mechanism of capitalist growth is that capital must extract (from nature and labour) more than it gives in return."

        result = nlp.get(document)
        assert isinstance(result, Sentiment)
        print("\n---")
        print("text: " + document)
        print(result)
        print(f"sentiment.positive: {result.positive}")
        print(f"sentiment.negative: {result.negative}")
        print(f"sentiment.neutral: {result.neutral}")

    def get_aspect_sentiment(self) -> None:
        """Get Aspect Sentiment."""
        nlp = AspectBasedSentimentAnalysis()

        examples = [
            ["1.) Instead of being at the back of the oven, the cord is attached at the front right side.","cord"],
            ["The pan I received was not in the same league as my old pan, new is cheap feeling and does not have a plate on the bottom.","pan"],
            ["The pan I received was not in the same league as my old pan, new is cheap feeling and does not have a plate on the bottom.","bottom"],
            ["They seem much more durable and less prone to staining, retaining their white properties for a much longer period of time.","durability"],
            ["It took some time to clean and maintain, but totally worth it!","clean"],
            ["this means that not only will the smallest burner heat up the pan, but it will also vertically heat up 1\" of the handle.","handle"]
        ]       
        for sentence, aspect in examples:
            start_time = datetime.now()
            print("\n---")
            result = nlp.get(sentence, aspect)
            assert isinstance(result, AspectSentiment)
            print(str(result))
            print("sentiment.text: " + sentence)
            print("sentiment.aspect: " + aspect)
            print("sentiment.mark: " + result.mark)
            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))

    def run(self) -> None:
        """Run."""
        self.get_sentiment()
        self.get_aspect_sentiment()


if __name__ == "__main__":
    TestSentimentAnalysis().run()