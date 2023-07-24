#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Text Emotion Detection."""

from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import pipeline


__all__ = (
    "Emotion",
    "EmotionDetectorT5",
    "EmotionDetectorRoberta",
)


@dataclass
class Emotion:
    """Emotion."""

    tag: str
    emoji: str


def get_emotion_emoji(tag: str) -> str:
    # Define the emojis corresponding to each sentiment
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
        "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(tag, "")


class EmotionDetectorT5:
    """Emotion Detector from T5 model."""

    # https://huggingface.co/mrm8488/t5-base-finetuned-emotion/

    # emotions = ["joy", "sad", "dis", "sup", "fea", "ang"]
    # emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def __init__(self) -> None:
        """Init Sentiment Analysis."""
        self.model_name = "mrm8488/t5-base-finetuned-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelWithLMHead.from_pretrained(self.model_name)

    def get(self, text: str) -> Emotion:
        """Check emotion from text string."""
        input_ids = self.tokenizer.encode(text + '</s>', return_tensors='pt')
        output = self.model.generate(input_ids=input_ids,
               max_length=2)
        dec = [self.tokenizer.decode(ids) for ids in output]
        emo = dec[0].replace("<pad>", "").strip()
        return Emotion(tag=emo, emoji=get_emotion_emoji(emo))


class EmotionDetectorRoberta:
    """Emotion Detector from Roberta."""

    # https://huggingface.co/SamLowe/roberta-base-go_emotions

    # emotions = [
    #     "admiration",
    #     "amusement",
    #     "anger",
    #     "annoyance",
    #     "approval",
    #     "caring",
    #     "confusion",
    #     "curiosity",
    #     "desire",
    #     "disappointment",
    #     "disapproval",
    #     "disgust",
    #     "embarrassment",
    #     "excitement",
    #     "fear",
    #     "gratitude",
    #     "grief",
    #     "joy",
    #     "love",
    #     "nervousness",
    #     "optimism",
    #     "pride",
    #     "realization",
    #     "relief",
    #     "remorse",
    #     "sadness",
    #     "surprise",
    #     "neutral",
    # ]

    def __init__(self) -> None:
        """Init."""
        self.model_name = "SamLowe/roberta-base-go_emotions"
        self.nlp = pipeline("sentiment-analysis", framework="pt", model=self.model_name)

    def get(self, text: str) -> Emotion:
        """Get Emotion from text str."""
        try:
            results = self.nlp(text)
        except RuntimeError as err:
            print(f"len(text) = {len(text)}")
            print(f"text: {text}")
            raise(err)

        data = {result['label']: result['score'] for result in results}
        tag, score = "", 0
        for key, value in data.items():
            if value > score:
                tag = key
                score = value
        return Emotion(
            tag=tag,
            emoji=get_emotion_emoji(tag=tag),
        )