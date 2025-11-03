#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pydantic import BaseModel, Field, ConfigDict


class TopicSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    """Description of topic during a time period"""

    # Title of the topic
    title: str
    # Relevant date
    date: str
    # Key Developments: major developments or trends
    key_developments: list[str]
    # Analysis
    description: str
    # Novelty compared to previous period
    novelty: str


class TopicSummaryList(BaseModel):
    model_config = ConfigDict(extra="allow")
    """Description of topic during a set of time period"""

    topic_summary_by_time_period: list[TopicSummary]


class PotentialImplications(BaseModel):
    """Potential Impact Analysis:
    - Examine the potential effects of this signal on various sectors, industries, and societal aspects.
    - Consider both short-term and long-term implications.
    - Analyze possible ripple effects and second-order consequences.
    """

    long_term_implications: list[str] = Field(default_factory=list)
    short_term_implications: list[str] = Field(default_factory=list)


class EvolutionScenario(BaseModel):
    """Evolution Scenarios
    - Describe potential ways this signal could develop or manifest in the future.
    - Consider various factors that could influence its trajectory.
    - Explore both optimistic and pessimistic scenarios.
    """

    optimistic_scenario_description: str = ""
    optimistic_scenario_points: list[str] = Field(default_factory=list)
    pessimistic_scenario_description: str = ""
    pessimistic_scenario_points: list[str] = Field(default_factory=list)


class TopicInterconnexions(BaseModel):
    """Interconnections and Synergies"""

    #  how this signal might interact with other current trends or emerging phenomena.
    interconnexions: list[str] = Field(default_factory=list)
    #  potential synergies or conflicts with existing systems or paradigms.
    ripple_effects: list[str] = Field(default_factory=list)


class Drivers(BaseModel):
    """Drivers and inhibitors"""

    # Factors that could accelerate or amplify a signal
    drivers: list[str] = Field(default_factory=list)
    # Potential barriers or resistances that might hinder its development.
    inhibitors: list[str] = Field(default_factory=list)


class SignalAnalysis(BaseModel):
    """Detailed analysis of topic evolution at a given time"""

    model_config = ConfigDict(extra="allow")

    potential_implications: PotentialImplications = Field(
        default_factory=PotentialImplications
    )
    evolution_scenario: EvolutionScenario = Field(
        default_factory=EvolutionScenario
    )
    topic_interconnexions: TopicInterconnexions = Field(
        default_factory=TopicInterconnexions
    )
    drivers_inhibitors: Drivers = Field(default_factory=Drivers)
