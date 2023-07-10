#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Common Methods."""

from pathlib import Path
from random import randint


def random_seed(a: int, b: int) -> int:
    """Random Seed."""
    return randint(a=a, b=b)

def path2str(fp: Path) -> str:
    """Get Path to str."""
    return str(fp)