# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from enum import Enum


class RewardModelType(Enum):
    task_validator = "task_validator_filter"
    accuracy_match = "keyword_match_penalty"
    sentence_match_penalty = "sentence_match_penalty"
    summary_relavance_match = "summary_relavance_match"
    twitter_content_relevance = "twitter_content_relevance"
    search_content_relevance = "search_content_relevance"
    performance_score = "performance_score"


class RewardScoringType(Enum):
    summary_relevance_score_template = "summary_relevance_score_template"
    link_content_relevance_template = "link_content_relevance_template"
    search_relevance_score_template = "search_relevance_score_template"
    performance_score_template = "performance_score_template"


@dataclass(frozen=True)
class DefaultRewardFrameworkConfig:
    """Reward framework default configuration.
    Note: All the weights should add up to 1.0.
    """

    summary_relevance_weight: float = 0.20
    twitter_content_weight: float = 0.45
    web_search_relavance_weight: float = 0.30
    performance_weight: float = 0.05


@dataclass(frozen=True)
class DefaultSummaryRelevanceWeightConfig:
    """Summary relevance weights configuration.
    Note: All the weights should add up to 1.0.
    """

    # Checks if summary is relevant to the prompt
    summary_weight: float = 0.60

    # Compares markdown descriptions to content of tweet or link title
    link_content_weight: float = 0.40
