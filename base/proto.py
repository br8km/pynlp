#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ProtoType for Data Structures."""

from typing import Any
from dataclasses import dataclass

"""
# Basic Model Structure for simulae Reality:

- Time -> Event


- Space

- Object -> Force

- Concept
- Force

# General Attributes include: movement, shape, credit_score

# TODO: thinking of more general ideas.

"""


@dataclass
class BaseDocObject:
    """Base Doc Object."""



@dataclass
class RealObject(BaseDocObject):
    """Real Object."""

    timestamp: float  # timestamp of this object
    space: tuple[float, float, float]  # space[x, y, z]
    attrs: dict[str, float]  # attributes, eg: movement, shape, etc.


@dataclass
class VirtualObject(BaseDocObject):
    """Virtual Object, eg: Concept, etc.."""

    timestamp: float  # timestamp of this object
    attrs: dict[str, float]  # attributes dictionary [str, vector]


@dataclass
class Relationship:
    """Relationship between Objects, eg: Force, Awareness, etc."""

    obj_from: str
    obj_to: str
    attrs: dict[str, float]


@dataclass
class Event:
    """Event for Objects."""

    timestamp: str
    obj: BaseDocObject
    attr: str
    change: float


@dataclass
class Force:
    """Force."""


@dataclass
class BaseItem:
    """Base Grammar."""

    parent: str  # Parent cls.obj.name

    name: str  # cls.obj.name
    ratio: float  # accurate ratio for this obj in all same level ones.


@dataclass
class Context(BaseItem):
    """Context, eg: Article, Post, Story, Book, Movie, Song, Scene, etc."""


@dataclass
class Kind(BaseItem):
    """Kind, eg: Knowedge, Information, etc."""


@dataclass
class Categroy(BaseItem):
    """Context Category."""


@dataclass
class Niche(BaseItem):
    """Context Category Niche."""


@dataclass
class Topic(BaseItem):
    """Context Category Niche Topic."""



@dataclass
class Intent:
    """Intent."""
    
    kind: str  # Knowedge, Information, etc.
    context: str  # Article, Post, Story, Book, Movie, Song, etc.
    category: str  
    niche: str

    lable: str
    rules: list[dict[str, Any]]


@dataclass
class Level:
    """Talk Trun Level."""
    

class CommonIntents:
    """Grammar Patterns."""

    intents: list[Intent]

    def load(self) -> list[Intent]:
        """Load list of Grammar Patterns."""
        raise NotImplementedError

    def update(self, intents: list[Intent]) -> bool:
        """Update one or list of Grammar patterns."""
        raise NotImplementedError

    def get(self,
            context: str,
            category: str,
            niche: str) -> list[Intent]:
        """Get Best Matched IntentPatterns."""
        if context and category and niche:
            return [x for x in self.intents if x.context==context and x.category==category and x.niche==niche]
        
        if context and category:
            return [x for x in self.intents if x.context==context and x.category==category]

        if context:
            return [x for x in self.intents if x.context==context]


        raise NotImplementedError
