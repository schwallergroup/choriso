"""Reaction utilities"""

import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rxn.chemutils.conversion import smiles_to_mol
from rxn.chemutils.miscellaneous import is_valid_smiles
from rxn.chemutils.reaction_equation import (
    canonicalize_compounds,
    merge_reactants_and_agents,
    sort_compounds,
)
from rxn.chemutils.reaction_smiles import ReactionFormat, parse_reaction_smiles, to_reaction_smiles
from rxn.chemutils.utils import remove_atom_mapping


def canonical_rxn(rxn_smi):
    """Canonicalize reaction SMILES. It returns Invalid SMILES if the
    SMILES is not correct. The function handles fragment bonds (containing '~')

    Args:
        rxn_smi: str, reaction SMILES to be canonicalized

    Out:
        new_reaction_smiles: str, canonicalized and cleaned reaction SMILES, including fragment bonds
    """
    try:
        # set reaction type to use fragment bonds (~)
        rxn_type = ReactionFormat.STANDARD_WITH_TILDE

        # first remove isotopes
        rxn_mol = AllChem.ReactionFromSmarts(rxn_smi)

        # Iterate over reactants, agents, and products
        for mol_list in [rxn_mol.GetReactants(), rxn_mol.GetAgents(), rxn_mol.GetProducts()]:
            for mol in mol_list:
                for atom in mol.GetAtoms():
                    # Set isotope number to 0
                    atom.SetIsotope(0)

        rxn = AllChem.ReactionToSmiles(rxn_mol)

        # parse full reaction SMILES
        ReactEq = parse_reaction_smiles(rxn, rxn_type)

        # Standard reaction: canonicalize reaction and sort compounds
        std_rxn = sort_compounds(canonicalize_compounds(merge_reactants_and_agents(ReactEq)))

        # Create final reaction SMILES
        rxn = to_reaction_smiles(std_rxn, rxn_type)

        return rxn

    except:
        return "Invalid SMILES"


def merge_reagents(rxn_smi):
    """set reaction type to use fragment bonds (~)"""

    rxn_type = ReactionFormat.STANDARD_WITH_TILDE
    rxn_eq = parse_reaction_smiles(rxn_smi, rxn_type)
    merged_eq = merge_reactants_and_agents(rxn_eq)
    return to_reaction_smiles(merged_eq, rxn_type)


def join_additives(row):
    """
    This function controls how the final reaction will look like.

    - Joint reagent, solvent and catalyst to the original reaction SMILES

    """
    additives_smiles = ["reagent_SMILES", "solvent_SMILES", "catalyst_SMILES"]

    # Aggregate all in a single string
    new_rxn = ".".join(row[list(additives_smiles)])
    # Remove `empty`
    new_rxn = re.sub("empty", "", new_rxn)
    # Add to simple rxn
    new_rxn += "." + row["rxn_smiles"]
    # Remove extra dots
    new_rxn = re.sub(r"\.+", ".", new_rxn)
    # Remove trailing dots
    new_rxn = new_rxn.strip(".")
    # return complete reaction
    return new_rxn


def is_reaction_valid(rxn):
    """check: molecules in reactions are valid smiles"""

    # Remove extended smiles appendix
    rxn = re.sub(r" +\|f:\d\..*\|", "", rxn)
    # Remove quotes
    rxn = re.sub(r"\"", "", rxn)

    mols = re.sub(">>", ".", rxn).split(".")
    return np.all(list(map(lambda m: is_valid_smiles(m, check_valence=False), mols)))


def has_more_than_max_tokens(rxn, rxnmapper):
    """remove reactions that are more than 512 tokens long"""

    tokens = rxnmapper.tokenizer.batch_encode_plus([rxn])["input_ids"][0]
    return len(tokens) >= 512


def canonical_smiles_mol(smi):
    """Convert molecular smiles into its canonical smiles"""

    smi = remove_atom_mapping(smi)
    return Chem.MolToSmiles(smiles_to_mol(smi, sanitize=False))
