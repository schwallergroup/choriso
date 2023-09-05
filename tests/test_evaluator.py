"""Test model evaluator pipeline."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from choriso.metrics.metrics.selectivity import Evaluator


@pytest.mark.skip(reason="dude be takin ages")
def test_flags():
    "Test if Evaluator is flagging reactions correctly"

    # get truth
    df_truth = pd.read_csv("tests/test_df_truth.csv")

    # use Evaluator to map and compute metrics
    ev = Evaluator("tests/test_df.csv", mapping=True, save=False)

    ev.compute_metrics(chemistry=True)

    # save processed df and compare to truth
    df = ev.file

    assert_frame_equal(df, df_truth)


@pytest.mark.skip(reason="dude be takin ages")
def test_score():
    """Test if metrics are computed correctly"""
    ev_truth = Evaluator("tests/test_df_truth.csv", mapping=False, save=False)

    ev_truth.compute_metrics(chemistry=True)

    # get truth
    ev = Evaluator("tests/test_df.csv", mapping=True, save=False)

    ev.compute_metrics(chemistry=True)

    assert ev.metrics == ev_truth.metrics
