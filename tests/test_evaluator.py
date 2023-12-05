"""Test model evaluator pipeline."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from choriso.metrics.selectivity import flag_regio_problem, flag_stereo_problem


# Need to rewrite this test. Evaluator no longer there
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


# Need to rewrite this test. Evaluator no longer there
@pytest.mark.skip(reason="dude be takin ages")
def test_score():
    """Test if metrics are computed correctly"""
    ev_truth = Evaluator("tests/test_df_truth.csv", mapping=False, save=False)

    ev_truth.compute_metrics(chemistry=True)

    # get truth
    ev = Evaluator("tests/test_df.csv", mapping=True, save=False)

    ev.compute_metrics(chemistry=True)

    assert ev.metrics == ev_truth.metrics


def test_flag_regio():
    """Test if regioselectivity is flagged correctly"""

    rxn = "BrCc1ccccc1.C1CCOC1.C=CC(O)CO.[H-].[Na+]>>C=CC(O)COCc1ccccc1"

    assert flag_regio_problem(rxn) == True


def test_flag_stereo():
    """Test if stereoselectivity is flagged correctly"""

    rxn = "C=C(NC(C)=O)c1ccc(OC)cc1.ClCCl.[H][H].[Rh+]>>COc1ccc([C@@H](C)NC(C)=O)cc1"

    assert flag_stereo_problem(rxn) == True
