"""Test data preprocessing functions"""

import pytest

from choriso.data.processing.preproc import *


#skip this test
@pytest.mark.skip(reason="not implemented yet")
def test_get_structures_from_name():
    """Test get_structures_from_name, text to SMILES function.

    Check first if Leadmine is available
    """

    if leadmine_flag:
        assert get_structures_from_name("ethanol") == "C(C)O"
        assert get_structures_from_name("hydrogen") == "[H][H]"
        assert get_structures_from_name("10percent pd/c") == "[Pd]"
    else:
        assert get_structures_from_name("hydrogen") == 0


def test_preprocess_additives():
    """Test preprocess additives function, extracting SMILES to name function
    for each column on test dataset."""

    if leadmine_flag:
        true = pd.read_csv("data/test/processed_leadmine.csv", sep="\t")
        processed = preprocess_additives("data/test/", "data_from_CJHIF_utf8")

        assert processed.iloc[0].equals(true.iloc[0])
    else:
        assert get_structures_from_name("hydrogen") == 0


def test_get_full_reaction_smiles():
    """Test get_full_reaction_smiles, join extracted SMILES"""

    true = pd.read_csv("data/test/full_processed.csv", sep="\t")
    raw = pd.read_csv("data/test/processed_leadmine.csv", sep="\t")
    raw["reagent_SMILES"] = raw["reagent_SMILES"].apply(lambda x: str(x))
    raw["catalyst_SMILES"] = raw["catalyst_SMILES"].apply(lambda x: str(x))
    raw["solvent_SMILES"] = raw["solvent_SMILES"].apply(lambda x: str(x))
    processed = get_full_reaction_smiles(raw)

    assert true.loc[0, "full_reaction_smiles"] == processed.loc[0, "full_reaction_smiles"]


def test_canonicalize_filter_reaction():
    """Test canonicalize_filter_reaction, canonicalize and filter
    duplicated SMILES"""

    true = pd.read_csv("data/test/full_processed.csv", sep="\t")
    raw = pd.read_csv("data/test/full_processed.csv", sep="\t").iloc[:, :-1]
    proc = canonicalize_filter_reaction(raw, "full_reaction_smiles")

    assert true.loc[0, "canonic_rxn"] == proc.loc[0, "canonic_rxn"]
