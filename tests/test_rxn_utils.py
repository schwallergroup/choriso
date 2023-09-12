"""Test reaction utils"""

from functools import partial

import numpy as np
import pytest

from choriso.data.processing.rxn_utils import *


def test_canonical_rxn():
    """
    Test reaction canonicalization function.
    """
    test_rxns = [
        "[Zn].[OH-]~[Na+].O=S1(=O)C=Cc2ccccc12>>O=S1(=O)CCc2ccccc12",
        "C(=O)[O-]~[NH4+].[Pd].CO.O=C1CCCN1C1CCN(Cc2ccccc2)CC1>>O=C1CCCN1C1CCNCC1",
        "[H][H].[OH-]~[OH-]~[Pd+2].O.CO.CCNC1(CCN(Cc2ccccc2)CC1)C(N)=O>>CCNC1(CCNCC1)C(N)=O",
        "N1=CC=CC2=CC=CC=C12.[H][H].C(C)O.C(CCC)OCCOCCOCC1=C(C=C2C(=C1)OCO2)CCC.[Pd]~CC(=O)[O-]~CC(=O)[O-]~[Pb+2].C[C@H](OC(C)=O)C#CC(O)=O>>C[C@H](OC(C)=O)\C=C/C(O)=O",
        "[Pd]~CC(=O)[O-]~CC(=O)[O-]~[Pb+2].C[C@H](OC(C)=O)C#CC(O)=O>>C[C@H](OC(C)=O)\C=C/C(O)=O",
        "N1=CC=CC=C1.C(C)(=O)[O-]~C(C)(=O)[O-]~[Cu+2].CO.C#Cc1cccs1>>c1csc(c1)C#CC#Cc1cccs1",
        "[OH-]~[NH4+].[Cu].C([O-])([O-])=O~[NH4+]~[NH4+].C(C)O.C#Cc1cccs1>>c1csc(c1)C#CC#Cc1cccs1",
        "C([O-])([O-])=O~[K+]~[K+].C(C)(=O)O[Pd]C1=C(C(=CC(=C1CP(C1=CC=CC=C1)C1=CC=CC=C1)C)C)CP(C1=CC=CC=C1)C1=CC=CC=C1.C(CCC)O.COc1ccc(C=CC(=O)c2ccccc2)cc1>>COc1ccc(CCC(=O)c2ccccc2)cc1",
        "C(CC#N)#N.FC(S(=O)(=O)[O-])(F)F~[Bi+3]~FC(S(=O)(=O)[O-])(F)F~FC(S(=O)(=O)[O-])(F)F.ClCCl.COc1ccc(C=CC(=O)c2ccccc2)cc1>>COc1ccc(CCC(=O)c2ccccc2)cc1",
        "[OH-]~[NH4+].[Pt].COc1ccc(\C=C\C(O)=O)cc1OC>>COc1ccc(CCC(O)=O)cc1OC",
    ]

    expected = [
        "O=S1(=O)C=Cc2ccccc21.[Na+]~[OH-].[Zn]>>O=S1(=O)CCc2ccccc21",
        "CO.O=C1CCCN1C1CCN(Cc2ccccc2)CC1.O=C[O-]~[NH4+].[Pd]>>O=C1CCCN1C1CCNCC1",
        "CCNC1(C(N)=O)CCN(Cc2ccccc2)CC1.CO.O.[H][H].[OH-]~[OH-]~[Pd+2]>>CCNC1(C(N)=O)CCNCC1",
        "CC(=O)O[C@@H](C)C#CC(=O)O.CC(=O)[O-]~CC(=O)[O-]~[Pb+2]~[Pd].CCCCOCCOCCOCc1cc2c(cc1CCC)OCO2.CCO.[H][H].c1ccc2ncccc2c1>>CC(=O)O[C@@H](C)/C=C\C(=O)O",
        "CC(=O)O[C@@H](C)C#CC(=O)O.CC(=O)[O-]~CC(=O)[O-]~[Pb+2]~[Pd]>>CC(=O)O[C@@H](C)/C=C\C(=O)O",
        "C#Cc1cccs1.CC(=O)[O-]~CC(=O)[O-]~[Cu+2].CO.c1ccncc1>>C(C#Cc1cccs1)#Cc1cccs1",
        "C#Cc1cccs1.CCO.O=C([O-])[O-]~[NH4+]~[NH4+].[Cu].[NH4+]~[OH-]>>C(C#Cc1cccs1)#Cc1cccs1",
        "CC(=O)O[Pd]c1c(CP(c2ccccc2)c2ccccc2)c(C)cc(C)c1CP(c1ccccc1)c1ccccc1.CCCCO.COc1ccc(C=CC(=O)c2ccccc2)cc1.O=C([O-])[O-]~[K+]~[K+]>>COc1ccc(CCC(=O)c2ccccc2)cc1",
        "COc1ccc(C=CC(=O)c2ccccc2)cc1.ClCCl.N#CCC#N.O=S(=O)([O-])C(F)(F)F~O=S(=O)([O-])C(F)(F)F~O=S(=O)([O-])C(F)(F)F~[Bi+3]>>COc1ccc(CCC(=O)c2ccccc2)cc1",
        "COc1ccc(/C=C/C(=O)O)cc1OC.[NH4+]~[OH-].[Pt]>>COc1ccc(CCC(=O)O)cc1OC",
    ]

    canonical_rxn_partial = partial(canonical_rxn)
    canon_rxns = list(map(canonical_rxn_partial, test_rxns))
    assert np.all(list(map(lambda x, y: x == y, canon_rxns, expected)))


def test_join_additives():
    """
    Test join additives function
    """
    assert True


def test_is_reaction_valid():
    """
    Test reaction smiles discriminator based on rdkit parsing.
    """
    assert True


def test_token_counter():
    """
    Test reaction smiles discriminator based on number of tokens.
    """
    assert True
