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


def flag_regio_problem(rxn, template=None):
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


def flag_stereo_problem(template=None, rxn_smiles=None):
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


class StereoCleaner:
    """Class for checking stereochemistry in reactions
    
    """
    def __init__(
        self,
        mapped_rxn: Union[str, List[str]],
    ) -> None:
        """
        Initialise the StereoCleaner class. The input can be a rxn
        SMILES or a list of rxn SMILES.
        """
        self.mapped_rxn = mapped_rxn
        self.correct_rxn: list[bool] = []
        self.not_sanitize: list = []

        assert isinstance(self.mapped_rxn, (str, list)), "Invalid input type"

        # assert all(">>" in rxn for rxn in self.mapped_rxn), "Invalid rxn SMILES"

        if isinstance(self.mapped_rxn, str):
            self.mapped_rxn = [self.mapped_rxn]

    def is_correct(self) -> Union[bool, List[bool]]:
        """Check if reactions are correct"""
        for idx, (reactants, products) in enumerate(self.iter_mol()):
            atom_creation = self.check_atom_creation(reactants, products)
            chiral_transfer = self.check_unreacted_centers(reactants, products)
            uncomplete_rxn = self.uncomplete_rxn(reactants, products)

        return chiral_transfer and atom_creation and uncomplete_rxn

    def check_unreacted_centers(self, reactants: Chem.Mol, products: Chem.Mol) -> bool:
        """Check if chirality is conserved in unreacted centers"""
        reactant = self.combine_mols(reactants)
        product = self.combine_mols(products)
        try:
            Chem.SanitizeMol(reactant)
            Chem.SanitizeMol(product)
        except Chem.MolSanitizeException:
            warnings.warn("!!!!Molecule could not be sanitized!!!!")
            return True

        reactant_chiral_dict = dict(Chem.FindMolChiralCenters(reactant))
        product_chiral_dict = dict(Chem.FindMolChiralCenters(product))

        # if not reactant_chiral_dict and product_chiral_dict:
        #     warnings.warn("Chiral centers appear in products without chiral centers in reactants")
        #     return False

        if not reactant_chiral_dict and not product_chiral_dict:
            return True

        for atom_idx, chirality in reactant_chiral_dict.items():
            chiral_atom = reactant.GetAtomWithIdx(atom_idx)
            chiral_atom_map = chiral_atom.GetAtomMapNum()
            if chiral_atom_map:  # chiral center mapped
                # get neibouring atoms
                neighbours = chiral_atom.GetNeighbors()
                neighbour_map = [neighbour.GetAtomMapNum() for neighbour in neighbours]
                if not all(neighbour_map):
                    warnings.warn(f"Chiral center {chiral_atom_map} is not fully mapped")
                    continue
                # get chiral center in product
                list_product_map = [atom.GetAtomMapNum() for atom in product.GetAtoms()]
                if chiral_atom_map not in list_product_map:
                    warnings.warn(
                        f"Chiral center {chiral_atom_map} in reactant is not present in product"
                    )
                    continue
                product_chiral_atom = product.GetAtomWithIdx(
                    list_product_map.index(chiral_atom_map)
                )
                # get neibouring atoms
                product_neighbours = product_chiral_atom.GetNeighbors()
                product_neighbour_map = [
                    neighbour.GetAtomMapNum() for neighbour in product_neighbours
                ]
                if not all(product_neighbour_map):
                    warnings.warn(f"Chiral center {chiral_atom_map} in product is not fully mapped")
                    continue

                hybridization_reactant = chiral_atom.GetHybridization()
                hybridization_product = product_chiral_atom.GetHybridization()
                if hybridization_reactant and hybridization_product:
                    if (hybridization_reactant == "SP3") and (hybridization_product == "SP2"):
                        # example : oxidation of chiral alcohol
                        warnings.warn("Changed in Hybridization, ignore")
                        continue
                # check chirality
                if product_chiral_atom.GetIdx() in product_chiral_dict.keys():
                    prod_chirality = product_chiral_dict[product_chiral_atom.GetIdx()]
                    if chirality != prod_chirality:
                        warnings.warn(
                            f"Chiral center {chiral_atom_map} has different chirality in product"
                        )
                        return False
        return True

    def check_atom_creation(self, reactants: List[Chem.Mol], products: List[Chem.Mol]) -> bool:
        """Check if any atom in product was not initially in the reactants"""
        reactant = self.combine_mols(reactants)
        product = self.combine_mols(products)

        set_reactants = set()
        set_products = set()

        for atom in reactant.GetAtoms():
            set_reactants.add(atom.GetSymbol())

        for atom in product.GetAtoms():
            set_products.add(atom.GetSymbol())

        if set_products - set_reactants:
            warnings.warn("Atom creation")
            return False
        return True

    def uncomplete_rxn(self, reactants: List[Chem.Mol], products: List[Chem.Mol]) -> bool:
        """Check if there is only one molecule in the reactants"""
        if (len(reactants) == 1) and (len(products) == 1):
            if Chem.MolToSmiles(reactants[0]) == Chem.MolToSmiles(products[0]):
                warnings.warn("Unreacted molecule")
                return False
            # get the dict of atoms in reactants and product
            reactant = reactants[0]
            product = products[0]
            reactant_dict = defaultdict(lambda: 0)
            product_dict = defaultdict(lambda: 0)
            print(reactant_dict)
            print(product_dict)
            for atom in reactant.GetAtoms():
                reactant_dict[atom.GetSymbol()] += 1
            for atom in product.GetAtoms():
                product_dict[atom.GetSymbol()] += 1
            # check if the number of atoms is the same
            if reactant_dict == product_dict:
                # This is a rearrangement
                return True
            else:
                return False

        return True

    def iter_mol(self) -> Iterator[Tuple[List[Chem.Mol], List[Chem.Mol]]]:
        """Iterate over the reactants and products"""
        for mapped_rxn in self.mapped_rxn:
            rxn = AllChem.ReactionFromSmarts(mapped_rxn)
            yield rxn.GetReactants(), rxn.GetProducts()

    @staticmethod
    def combine_mols(mols: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """Combine a list of molecules into one"""
        if len(mols) == 0:
            return None
        elif len(mols) == 1:
            return mols[0]

        combined_molecule = mols[0]
        for mol in mols[1:]:
            if mol is not None:
                combined_molecule = Chem.CombineMols(combined_molecule, mol)

        return combined_molecule
