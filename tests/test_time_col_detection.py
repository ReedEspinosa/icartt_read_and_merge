"""Regression tests for _find_datelike_cols: the header-units veto/rescue
must work on RAW column names and on names carrying the instrument-title
prefix (the multi-instrument merge prefixes columns BEFORE time parsing;
an exact-key units lookup silently disabled the veto there, and AMS
'*_lt_1um_*' species were consumed as Local Time with one of them picked
as the file's time base -- SEAC4RS 2013, found 2026-09-03)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icartt_read_and_merge import icartt_read_and_merge as m  # noqa: E402


@pytest.fixture()
def ams_like_ict(tmp_path):
    """Minimal ICARTT file shaped like SEAC4RS AMS-60S."""
    header = [
        "10, 1001",
        "Jimenez, Jose L",
        "U Colorado",
        "AMS test",
        "SEAC4RS",
        "1, 1",
        "2013, 08, 06, 2014, 01, 01",
        "60.0",
        "AMS_Starttime, seconds_past_midnight, start time",
        "3",
        "AMS_Stoptime, seconds_past_midnight, stop time",
        "OA_lt_1um_AMS_60s, ugsm-3, organic aerosol mass",
        "Chloride_lt_1um_AMS_60s, ugsm-3, chloride mass",
    ]
    # header line 1 says n_header=10 -> keep consistent: recount
    lines = header[1:]  # drop placeholder first line
    n_header = len(lines) + 2  # +1 for the count line, +1 for column names
    body = ["AMS_Starttime,AMS_Stoptime,OA_lt_1um_AMS_60s,Chloride_lt_1um_AMS_60s",
            "64860,64920,1.5,0.02",
            "64920,64980,1.7,0.03"]
    txt = "\n".join([f"{n_header}, 1001"] + lines + body) + "\n"
    p = tmp_path / "SEAC4RS-AMS-60S_DC8_20130806_R1.ict"
    p.write_text(txt)
    return str(p)


def _frame():
    return pd.DataFrame({
        "AMS_Starttime": [64860.0, 64920.0],
        "AMS_Stoptime": [64920.0, 64980.0],
        "OA_lt_1um_AMS_60s": [1.5, 1.7],
        "Chloride_lt_1um_AMS_60s": [0.02, 0.03],
    })


def test_raw_names_species_vetoed_startstop_rescued(ams_like_ict):
    df, times, _ = m._find_datelike_cols(_frame(), ams_like_ict)
    assert sorted(times) == ["ams_starttime", "ams_stoptime"]
    assert "OA_lt_1um_AMS_60s" in df.columns


def test_prefixed_names_species_vetoed_startstop_rescued(ams_like_ict):
    pref = "Non-refractory__chemical_speciated_mass_"
    df = _frame().rename(columns=lambda c: pref + c)
    df, times, _ = m._find_datelike_cols(df, ams_like_ict)
    assert sorted(times) == [pref.lower() + "ams_starttime",
                             pref.lower() + "ams_stoptime"]
    assert pref + "OA_lt_1um_AMS_60s" in df.columns
    assert pref + "Chloride_lt_1um_AMS_60s" in df.columns
