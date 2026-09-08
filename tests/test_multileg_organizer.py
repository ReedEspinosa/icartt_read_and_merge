"""Regression tests for _organize_standard_and_multileg_flights: some
archives carry BOTH a combined file and per-leg files for the same flight
(ACTIVATE-LARGE-CCN_HU25_20220608_R0.ict alongside _R0_L1/_R0_L2 — found
2026-09-08 while merging ACTIVATE 2022). The organizer must prefer the
combined file instead of crashing with AttributeError('str'.append)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icartt_read_and_merge import icartt_read_and_merge as m  # noqa: E402

COMBINED = "ACTIVATE-LARGE-CCN_HU25_20220608_R0.ict"
LEG1 = "ACTIVATE-LARGE-CCN_HU25_20220608_R0_L1.ict"
LEG2 = "ACTIVATE-LARGE-CCN_HU25_20220608_R0_L2.ict"
OTHER_L1 = "ACTIVATE-LARGE-SMPS_HU25_20220608_R1_L1.ict"
OTHER_L2 = "ACTIVATE-LARGE-SMPS_HU25_20220608_R1_L2.ict"


def organize(files):
    return m._organize_standard_and_multileg_flights({"ICARTT_FILES": files})


def test_pure_legs_are_grouped():
    flights = organize([OTHER_L1, OTHER_L2])
    assert flights == {
        "ACTIVATE-LARGE-SMPS_HU25_20220608_R1.ict": [OTHER_L1, OTHER_L2]}


def test_combined_first_then_legs_prefers_combined():
    flights = organize([COMBINED, LEG1, LEG2])
    assert flights == {COMBINED: COMBINED}


def test_legs_first_then_combined_prefers_combined():
    flights = organize([LEG1, LEG2, COMBINED])
    assert flights == {COMBINED: COMBINED}


def test_mixed_flight_set_keeps_other_instruments():
    flights = organize([LEG1, COMBINED, OTHER_L1, OTHER_L2])
    assert flights[COMBINED] == COMBINED
    assert flights["ACTIVATE-LARGE-SMPS_HU25_20220608_R1.ict"] == [
        OTHER_L1, OTHER_L2]
