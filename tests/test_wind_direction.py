"""Wind direction must be vector-averaged (circular), not scalar-averaged.

align2master_timeline resamples every ICARTT file onto the master timeline.
A compass bearing cannot be averaged or linearly interpolated as a plain
scalar: near the 0/360 discontinuity the mean of 350 and 10 degrees is 0/360,
not 180. These tests pin the circular (unit-vector) handling in both the
averaging branch (native finer than the target step) and the interpolation
branch (native coarser).
"""

import numpy as np
import pandas as pd
import pytest

from icartt_read_and_merge.icartt_read_and_merge import (
    align2master_timeline as a2m,
    _wind_direction_cols,
)

START, END = "2021-05-13 00:00:00", "2021-05-13 00:10:00"


def _circmean_deg(series):
    r = np.deg2rad(series.dropna().to_numpy())
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0


def _frame(values, freq, col="WDIR_deg", start="2021-05-13 00:01:00"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    df = pd.DataFrame({col: np.asarray(values, float)}, index=idx)
    df.index.name = "datetime"
    return df


def test_detection_matches_wind_direction_only():
    cols = ["ACTIVATE-WINDS_WDIR_deg", "Wind_Direction", "WindDir", "wind_dir",
            "Latitude_deg", "Longitude", "True_Heading", "Pitch_deg", "Roll_deg"]
    assert _wind_direction_cols(cols) == [
        "ACTIVATE-WINDS_WDIR_deg", "Wind_Direction", "WindDir", "wind_dir"]


def test_averaging_branch_circular_across_wrap():
    # 20 Hz alternating 350/10 deg -> circular mean 0/360, never 180
    df = _frame(np.where(np.arange(1200) % 2 == 0, 350.0, 10.0), "50ms")
    out = a2m(df, START, END, step_S=1, datetime_index=True)["WDIR_deg"].dropna()
    assert not ((out > 90) & (out < 270)).any()          # no scalar-mean 180
    d = np.minimum(out % 360, 360 - (out % 360))
    assert (d < 15).all()                                # all hug the wrap


def test_averaging_branch_tracks_smooth_wrap():
    # smooth ramp 355 -> 5 deg at 20 Hz; per-second output tracks the truth
    n = 1200 * 3
    truth = (355 + np.linspace(0, 10, n)) % 360
    out = a2m(_frame(truth, "50ms"), START, END, step_S=1,
              datetime_index=True)["WDIR_deg"].dropna()
    ref = pd.Series(truth, index=pd.date_range(
        "2021-05-13 00:01:00", periods=n, freq="50ms", tz="UTC")
        ).groupby(pd.Grouper(freq="1s")).apply(_circmean_deg).dropna()
    a, b = out.align(ref, join="inner")
    err = np.abs((a - b + 180) % 360 - 180)
    assert err.max() < 3.0
    assert not ((out > 90) & (out < 270)).any()


def test_interpolation_branch_no_wrap_dive():
    # coarse 5 s direction crossing 360; interpolation must not dip toward 180
    df = _frame(np.linspace(340, 380, 12) % 360, "5s", col="Wind_Direction")
    out = a2m(df, START, END, step_S=1, datetime_index=True)["Wind_Direction"].dropna()
    assert not ((out > 90) & (out < 270)).any()


def test_steady_direction_unchanged():
    out = a2m(_frame(np.full(1200, 275.0), "50ms"), START, END, step_S=1,
              datetime_index=True)["WDIR_deg"].dropna()
    assert np.allclose(out.to_numpy(), 275.0, atol=1e-6)


def test_non_direction_column_untouched():
    # a plain scalar column resamples identically whether or not a wind-dir
    # column is present alongside it
    vals = np.linspace(0.0, 100.0, 1200)
    wind = np.full(1200, 275.0)
    df = _frame(vals, "50ms", col="Temperature_C")
    df["WDIR_deg"] = wind
    with_wind = a2m(df, START, END, step_S=1, datetime_index=True)["Temperature_C"]
    without = a2m(_frame(vals, "50ms", col="Temperature_C"), START, END,
                  step_S=1, datetime_index=True)["Temperature_C"]
    pd.testing.assert_series_equal(with_wind.dropna(), without.dropna())
