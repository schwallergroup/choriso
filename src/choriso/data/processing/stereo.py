"""Module with code to filter and flag reactions based on stereo and 
regioselectivity."""
import signal
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction
from transformers import logging


def flag_stereoalchemy(rxn):
    """Flag reactions that contain stereochemistry ('@') in products but not in reactants

    Args:
        rxn: str, reaction SMILES

    Out:
        bool, True if the reaction has stereochemistry issues, False otherwise

    """
    reacs = rxn.split(">>")[0].split(".")
    prods = rxn.split(">>")[1].split(".")

    if any("@" in prod for prod in prods):
        if not any("@" in reac for reac in reacs):
            return True
        else:
            return False

    else:
        return False


def has_carbon(smiles):
    """Check if a SMILES contains carbon."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]"))


def remove_chiral_centers(reaction_smiles):
    """Remove chiral centers from a reaction SMILES.

    Args:
        reaction_smiles: str, reaction SMILES

    Out:
        str, reaction SMILES without chiral centers

    """
    # Parse the reaction SMILES
    rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)

    # Function to remove chiral centers from a molecule
    def remove_chirality(mol):
        "Remove any stereochemical information from a molecule"
        Chem.RemoveStereochemistry(mol)
        return mol

    # Process reactants and products
    for side in [rxn.GetReactants(), rxn.GetProducts()]:
        for mol in side:
            remove_chirality(mol)

    # Return the modified reaction SMILES
    return AllChem.ReactionToSmiles(rxn)

