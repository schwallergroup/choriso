"""Module with code to filter and flag reactions based on stereo and 
regioselectivity."""
import signal
import warnings
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple, Union

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
        "Remove chiral centers from a molecule"
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        return mol

    # Process reactants and products
    for side in [rxn.GetReactants(), rxn.GetProducts()]:
        for mol in side:
            remove_chirality(mol)

    # Return the modified reaction SMILES
    return AllChem.ReactionToSmiles(rxn)


def aam_from_smiles(list_rxn_smiles):
    """Get attention guided atom maps from a list of reaction SMILES.
    Args:
        list_rxn_smiles: list, reaction SMILES
    Out:
        out: list, attention guided atom maps
    """
    rxn_mapper = RXNMapper()
    out = []
    for i, rxn in enumerate(list_rxn_smiles):
        try:
            out.append(rxn_mapper.get_attention_guided_atom_maps([rxn])[0])
        except:
            out.append({"confidence": "", "mapped_rxn": ""})
            print("Issue with reaction", i, rxn)
    return out


def template_smarts_from_mapped_smiles(mapped_smiles, radius=0):
    """Get reaction template from mapped reaction SMILES. If mapping time
    exceeds 60 seconds, return False.

    Args:
        mapped_smiles: str, mapped reaction SMILES
        radius: int, radius of the reaction template

    Out:
        template: str, reaction template
    """

    def signal_handler(signum, frame):
        """Handle very long requests"""
        raise Exception("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(50)

    try:
        rxn = ChemicalReaction(mapped_smiles, clean_smiles=False)
        rxn.generate_reaction_template(radius)
        return rxn.canonical_template.smarts

    except:
        return False


def _flag_regio_problem(rxn, template=None):
    """Flag regioselectivity problems. For the moment only one-product
    reactions. The function extracts the reaction template (only reacting atoms) and then checks
    if the matching atoms in the product can generate several products.

    Args:
        rxn: str, reaction SMILES
        template: str (optional), reaction template (r=1). If not provided,
                  the function will try to extract it from the reaction SMILES.

    Out:
        bool, True if the reaction is regioselective, False otherwise
    """

    def _sanitize_filter_prods(prods):
        good = []
        for prod in prods:
            try:
                x = Chem.SanitizeMol(prod[0])
                good.append(Chem.MolToSmiles(prod[0]))
            except:
                pass
        return set(good)

    if template is not None:
        # extract rxn template
        map_rxn = aam_from_smiles([rxn])[0]["mapped_rxn"]
        template = template_smarts_from_mapped_smiles(map_rxn, radius=1)

    if template:
        products = rxn.split(">>")[1]

        if "@" in products:
            return False

        rxn_list = rxn.split(">>")[0].split(".")
        map_list = map_rxn.split(">>")[0].split(".")

        # compare lists and extract only the elements from rxn_list that are different from rxn_smiles_list
        reactants = [i for i in rxn_list if i not in map_list]

        # check if reactants generate several products
        reaction = AllChem.ReactionFromSmarts(template)
        reaction.Initialize()

        try:
            if len(reactants) == 2:
                r1 = Chem.MolFromSmiles(reactants[0])
                r2 = Chem.MolFromSmiles(reactants[1])

                mols = [(r1, r2), (r2, r1)]

                for reactants in mols:
                    new_products = reaction.RunReactants(reactants)
                    if new_products == ():
                        pass
                    else:
                        products = _sanitize_filter_prods(new_products)

            if len(reactants) == 1:
                r1 = Chem.MolFromSmiles(reactants[0])
                new_products = reaction.RunReactants((r1,))
                products = _sanitize_filter_prods(new_products)

            if len(products) == 1:
                return False
            elif len(products) > 1:
                return True
            else:
                return False

        except:
            return False
    else:
        return False


def _flag_stereo_problem(template=None, rxn_smiles=None):
    """Flag stereoselectivity problems.
    Args:
        template: str (optional), reaction template (r=0). If not provided,
                  the function will try to extract it from the reaction SMILES.
        rxn_smiles: str (optional), reaction SMILES.

    Out:
        bool, True if the reaction has stereoselectivity issues, False otherwise
    """

    def check_stereo(temp):
        """Check if the reaction template contains stereochemistry"""

        try:
            temp_prods = temp.split(">>")[1].split(".")
            # check if any of the strings in prods contain '@'
            if any("@" in prod for prod in temp_prods):
                return True

            else:
                return False
        except:
            return False

    if template is None and rxn_smiles is None:
        raise ValueError("Either template or rxn_smiles must be provided")

    if template is None:
        map_rxn = aam_from_smiles([rxn_smiles])[0]["mapped_rxn"]
        template = template_smarts_from_mapped_smiles(map_rxn, radius=1)
        return check_stereo(template)

    else:
        return check_stereo(template)
