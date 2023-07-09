#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test Emotion Detection."""

from datetime import datetime
from typing import Callable

from ..core.emotion import Emotion, EmotionDetectorT5, EmotionDetectorRoberta


class TestEmotionDetector:
    """Test Emotion Detector."""

    def run_test(self, func: Callable, text: str) -> None:
        """Run Test."""
        start_time = datetime.now()
        emo = func(text)
        assert isinstance(emo, Emotion)
        end_time = datetime.now()
        print("\ntext: " + text)
        print(type(emo), emo.tag)
        print('Duration: {}'.format(end_time - start_time))

    def run(self) -> None:
        """Run."""
        texts = [
            "i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute",
            "i have a feeling i kinda lost my best friend",
        ]
        nlp_t5 = EmotionDetectorT5()
        for text in texts:
            self.run_test(nlp_t5.get, text)

        nlp_rb = EmotionDetectorRoberta()
        for text in texts:
            self.run_test(nlp_rb.get, text)


if __name__ == "__main__":
    TestEmotionDetector().run()