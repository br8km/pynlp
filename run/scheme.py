# -*- coding: utf-8 -*-

"""Market Scheme."""

from dataclasses import dataclass, asdict
from dacite import from_dict


@dataclass
class CustomerIntent:
    """Customer Intent."""


class CustomerComplains(CustomerIntent):
    """Customer Pains, eg: frustration, anger, pain point, etc."""
    # Have Problem and don't know how to deal with it


class CustomerRequests(CustomerIntent):
    """Customer Requests, eg: solution, advice, resource, etc."""
    # Have problem and know some solution, ask for it or advice


class CustomerIdeas(CustomerIntent):
    """Customer Ideas, eg: tools, ways, alternatives, etc."""
    # Have problem and looking for solution or better ones


class CustomerMoneyTalks(CustomerIntent):
    """Customer Money Talks, eg: budget, pricing, spends, etc."""
    # Have problem and ideal solution, quote for price to get it



@dataclass
class AffiliteResponse:
    """Affiliate Response."""

    keyword: str

    anchor: str
    link: str

    templates: list[str]  # markdown|sections|spintax|placeholder

    # TODO: 
    # method -> template.replace.placeholder
    # method -> template.spinner


@dataclass
class AffiliateGuides(AffiliteResponse):
    """Affiliate guides, reviews to a problem solution."""



@dataclass
class AffiliateBenefits(AffiliteResponse):
    """Affiliate benefits, features to a problem solution."""


@dataclass
class AffiliateCompares(AffiliteResponse):
    """Affiliate comparisions to problem solutions."""


@dataclass
class AffiliatePricing(AffiliteResponse):
    """Affiliate pricing intro, coupon to a problem solution."""


@dataclass
class Audience:
    """Audience."""

    category: str
    niche: str

    nsfw: bool

    keywords: list[str]
    subreddits: list[str]

    # TODO: 
    # method -> user_count|user_active
    # method -> add|delete| subreddit|keyword


@dataclass
class Tracker:
    """Tracker.

        :keywords to tracking:
            - product category keywords, ie: "SEO tool", "keyword planner".
            - customer objectives, ie: "improve SEO", "get backlinks".
            - customer pain points, ie: "low domain rank", "no impressions".
            - your brand name.
            - competitor mentions, ie: "ahrefs", "moz", "semrush".
            - blog post topics, ie: "keyword difficulty", site speed".
    
    """

    audience: Audience
    intent: CustomerIntent
    keywords: list[str]