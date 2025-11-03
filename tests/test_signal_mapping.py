#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
from bertrend.llm_utils.llama3_client import map_signalanalysis_response
from bertrend.trend_analysis.data_structure import SignalAnalysis


def _validate(mapped: dict):
    obj = SignalAnalysis.model_validate(mapped)
    # Ensure required top blocks exist and are lists/strings
    assert isinstance(obj.potential_implications.short_term_implications, list)
    assert isinstance(obj.potential_implications.long_term_implications, list)
    assert isinstance(obj.evolution_scenario.optimistic_scenario_description, str)
    assert isinstance(obj.evolution_scenario.optimistic_scenario_points, list)
    assert isinstance(obj.topic_interconnexions.interconnexions, list)
    assert isinstance(obj.drivers_inhibitors.drivers, list)
    return obj


def test_variant_potential_impacts_mapping():
    raw = {
        "signal_analysis": {
            "potential_impacts": {
                "short_term": [{"sector": "Energy", "description": "Increased investment in renewable energy"}],
                "long_term": [{"sector": "Environment", "description": "Reduced greenhouse gas emissions"}],
            }
        }
    }
    mapped = map_signalanalysis_response(raw)
    obj = _validate(mapped)
    assert "Increased investment" in obj.potential_implications.short_term_implications[0]
    assert "Reduced greenhouse" in obj.potential_implications.long_term_implications[0]


def test_variant_development_scenarios_mapping():
    raw = {
        "signal_analysis": {
            "development_scenarios": {
                "optimistic": {
                    "description": "Rapid adoption of renewable energy",
                    "key_drivers": ["Government support"],
                },
                "pessimistic": {
                    "description": "Slow adoption of renewables",
                },
            }
        }
    }
    mapped = map_signalanalysis_response(raw)
    obj = _validate(mapped)
    assert "Rapid adoption" in obj.evolution_scenario.optimistic_scenario_description
    assert any("Government" in s for s in obj.evolution_scenario.optimistic_scenario_points)
    assert "Slow adoption" in obj.evolution_scenario.pessimistic_scenario_description


def test_variant_interconnections_and_drivers_mapping():
    raw = {
        "signal_analysis": {
            "interconnections_and_synergies": {
                "interactions": [
                    {"signal": "Climate change", "description": "Increased urgency"}
                ],
                "synergies": [
                    {"signal": "Energy efficiency", "description": "Adoption of efficient tech"}
                ],
            },
            "drivers_and_inhibitors": {
                "drivers": [
                    {"factor": "Government support", "description": "Policy incentives"}
                ],
                "inhibitors": [
                    {"factor": "High costs", "description": "Barriers"}
                ],
            },
        }
    }
    mapped = map_signalanalysis_response(raw)
    obj = _validate(mapped)
    assert any("Climate change" in s for s in obj.topic_interconnexions.interconnexions)
    assert any("Energy efficiency" in s for s in obj.topic_interconnexions.ripple_effects)
    assert any("Government support" in s for s in obj.drivers_inhibitors.drivers)
    assert any("High costs" in s for s in obj.drivers_inhibitors.inhibitors)


