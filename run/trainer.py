#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Trainer."""

from pathlib import Path

from spacy.language import Language
from spacy.cli.info import info
from spacy.cli.init_config import fill_config
from spacy.cli.train import train

from ..config import Config


class Trainer:
    """My Trainer.

        :Features:
            - Custom Training Args
            - Pause & Resume from source model files.
        
        :Examples:
            - https://github.com/explosion/projects/tree/v3/tutorials/textcat_goemotions
    """
    config = Config()

    def __init__(self,
                 model_name: str,
                 file_config: Path,
                 file_train: Path,
                 file_dev: Path,
                 output_path: Path,
                 ) -> None:
        """Init Trainer."""
        self.model_name = model_name
        self.file_config = file_config
        self.file_train = file_train
        self.file_dev = file_dev
        self.output_path = output_path

    def load_model(self) -> Language:
        """Load language model for train."""

    def info(self) -> None:
        """Show train.cli info."""
        print(info())

    def fill_config(self) -> bool:
        """Fill config from base-config."""
        file_base_config = self.config.dir_model / "base_config.cfg"
        fill_config(self.file_config, file_base_config)  # different order
        return self.file_config.is_file()

    def _train(self, overrides: dict) -> None:
        """Train."""
        train(config_path=self.file_config,
              output_path=self.output_path,
              overrides=overrides,
        )

    def start_train(self, stop: str) -> bool:
        """Start Train Process."""
        return self._train(
            overrides={
                "paths.train": self.file_train,
                "paths.dev": self.file_dev,
            }
        )

    def resume_train(self, component: str, source: str) -> bool:
        """Resume train process after paused."""
        # overrides
        # component -> factory|source -> exist.model.file -> best|last
        raise NotImplementedError


if __name__ == "__main__":
    Trainer()