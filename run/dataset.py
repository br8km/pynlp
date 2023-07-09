#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Emotion Dataset Processor."""

import json
from pprint import pprint
from pathlib import Path


import pandas as pd
from pandas import DataFrame

from ..inc.io import IO



class EmotionDataset:
    """Emotion Dataset."""

    dir_app = Path(__file__).parent
    dir_dat = dir_app / "dat"

    def goemotions_from_csv(self) -> None:
        """Process goemotions dataset."""
        dir_emo = self.dir_dat / "nlp.dataset" / "goemotions"

        files = [dir_emo / f"goemotions_{index}.csv" for index in range(1,4)]
        print(files[0].name)
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        # print(df.head())
        # print(df.__dict__)
        # remove example_very_unclear
        # generate new DataFrame with columns[text, emotion] only.
        # DataFrame.Text.Emotion
        emotions = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]

        data: list[dict] = []
        number, limit = 0, 0
        for index in df.index:
            text = df["text"][index]
            unclear = df["example_very_unclear"][index]
            if unclear:
                continue
            emotion = ""
            for emo in emotions:
                if df[emo][index] == 1:
                    emotion = emo
                    break
            print(index, text, unclear, emotion)
            data.append(
                {
                    "index": number,
                    "text": text,
                    "emotion": emotion,
                }
            )
            number += 1
            if limit and number >= limit:
                break

        file = dir_emo / "goemotions.json"
        IO.save_list_dict(file, data)
        assert file.is_file()

    def goemotions_from_json(self) -> None:
        """Goemotions."""
        dir_emo = self.dir_dat / "nlp.dataset" / "goemotions"
        file = dir_emo / "goemotions.json"

        data = IO.load_list_dict(file)
        print(len(data))
        for item in data[-5:]:
            print(json.dumps(item, indent=2))

            # pprint(item, indent=2, compact=True)

        
    def run(self) -> None:
        """Run."""
        self.goemotions_from_json()


if __name__ == "__main__":
    EmotionDataset().run()
