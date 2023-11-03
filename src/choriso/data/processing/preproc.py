"""Main preprocessing pipeline"""

import os
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd
import requests
from pandarallel import pandarallel
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rxn.chemutils.reaction_equation import (
    canonicalize_compounds,
    merge_reactants_and_agents,
    sort_compounds,
)
from rxn.chemutils.reaction_smiles import (
    ReactionFormat,
    parse_extended_reaction_smiles,
    to_reaction_smiles,
)
from tqdm.auto import tqdm

from choriso.data.processing import rxn_utils
from choriso.data.processing.custom_logging import print

pandarallel.initialize(progress_bar=True, nb_workers=22)

# Get correction dictionary (combination of the one obtained with Pyopsin + PubChem and manual correction)
try:
    # load general dictionary
    df_general_dict = pd.read_csv(
        "data/helper/cjhif_translation_table.tsv", sep="\t", usecols=["Compound", "pubchem"]
    ).fillna("empty_translation")
    general_dict = {row["Compound"]: row["pubchem"] for _, row in df_general_dict.iterrows()}

    # load manual correction dictionary
    df = pd.read_csv("data/helper/corrected_leadmine.csv", sep="\t", header=None, index_col=0)

    correct_dict = {row[0]: row[1].values[0] for row in df.iterrows()}

    # merge dictionaries
    full_dict = {**general_dict, **correct_dict}

    # replace nan values with "empty_translation"
    for key, value in full_dict.items():
        if type(value) == float:
            full_dict[key] = "empty_translation"

except:
    full_dict = {}
    print("Correction dictionary not available")


def download_raw_data(data_dir="data/raw/"):
    """Download the raw CJHIF dataset."""

    url = "https://drive.switch.ch/index.php/s/uthL9jTERVQJJoW/download"
    target_path = data_dir + "raw_cjhif.tar.gz"

    if not os.path.isfile(target_path):  # Only download if file doesn't already exist
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())

    # Decompress tar file
    with tarfile.open(data_dir + "raw_cjhif.tar.gz") as f:
        f.extractall(data_dir)

    # Substitute "§" with tabs in extracted file
    subprocess.call([f"sed -i -e 's/§/\t/g' {data_dir}data_from_CJHIF_utf8"], shell=True)

    # Load data
    raw_df = pd.read_csv(data_dir + "data_from_CJHIF_utf8", sep="\t")

    return 0


def download_processed_data(data_dir="data/processed/"):
    """Download processed data (after cleaning and atom mapping)."""

    # CHANGE THIS WITH THE CORRECTED CHORISO (CLEAN CJHIF + USPTO)
    base_url = "https://drive.switch.ch/index.php/s/VaSVBCiXrmzYzGD/download?path=%2F&files={}"

    print("Downloading processed datasets...")

    files = ["choriso.tar.gz", "uspto.tar.gz"]
    for fname in files:
        url = base_url.format(fname)

        target_path = data_dir + fname

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.isfile(target_path):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(response.raw.read())

        with tarfile.open(data_dir + fname) as f:
            f.extractall(data_dir)


def parse_entities(entity):
    """Return SMILES from detected text entity"""

    # If the name is empty, return the same to mark the column
    if entity == "empty":
        return "empty"

    # If entity in 'correct_dict', correct
    else:
        try:
            smiles = full_dict[entity]

            if smiles is not None:
                return smiles

            else:
                return "empty_translation"

        except KeyError:
            return "empty_translation"


def get_structures_from_name(names, format_bond=False):
    """Convert text with chemical structures to their corresponding SMILES
    using Pyopsin and Pubchem dictionary.

    Args:
        names: str or set, text containing chemical structures separated by the '|' character
        format_bond: bool, use '~' to represent chemical species from the same compound
                     e.g: [Na+]~[Cl-] instead of [Na+].[Cl-]

    Returns:
        structures: set, chemical entities SMILES from text

    """

    # split names by '|'
    names_list = names.split("|")

    # parse each entity
    structures = [parse_entities(name) for name in names_list]

    if format_bond:
        structures = [structure.replace(".", "~") for structure in structures]

    structures = ".".join(structures)

    return structures


def column_check(entry):
    """Check if translation was correct. A correct translation does not contain
    the 'empty_translation' string.

    Args:
        entry: str, text containing translated SMILES

    Returns:
        match: bool, True if translation was correct, False otherwise
    """

    smiles = entry.split(".")

    if "empty_translation" in smiles:
        return False
    else:
        return True


def preprocess_additives(data_dir, file_name, name="cjhif", logger=False):
    """First dataset preprocessing:
    Adapt additives information (solvents, catalysts, reagents).
        - Drop duplicates
        - Drop AMM and FG columns
        - Rename columns
        - Map additives' names to structures (SMILES) in new columns.
        - Drop rows where translation was faulty
        - Drop rows where reactants, catalyst and solvent columns are empty

    Args:
        data_dir: str, path to raw data
        file_name: str, name of the file containing raw data
        name: str, name of the dataset
        logger: bool, if True, log information about the preprocessing

    Returns:
        cjhif_no_empties: pd.DataFrame, preprocessed dataset
    """

    # Create df with raw data
    cjhif = (
        pd.read_csv(
            data_dir + file_name,
            sep="\t",
            header=None,
        )
        # Drop duplicate rows
        .drop_duplicates()
        # Fill NaN with empty strings
        .fillna("empty")
        # Drop columns 1 and 2
        .drop(columns=[1, 2], axis=1)
        # Rename columns
        .rename(columns={0: "rxn_smiles", 3: "reagent", 4: "solvent", 5: "catalyst", 6: "yield"})
    )

    ##DELETE THIS WHEN PUSHING
    cjhif = cjhif.sample(10000, random_state=33)
    # Map reagent text to SMILES
    print("Getting reagent SMILES")
    cjhif["reagent_SMILES"] = cjhif["reagent"].parallel_apply(get_structures_from_name)

    # Map solvent text to SMILES
    print("Getting solvent SMILES")
    cjhif["solvent_SMILES"] = cjhif["solvent"].parallel_apply(get_structures_from_name)

    # Map catalyst text to SMILES
    print("Getting catalyst SMILES")
    cjhif["catalyst_SMILES"] = cjhif["catalyst"].parallel_apply(get_structures_from_name)

    # Check if reagents and catalyst name have been correctly processed by Leadmine
    print("Checking reagent number")
    reagent_flag = cjhif["reagent_SMILES"].parallel_apply(lambda x: column_check(x))
    print("Checking catalyst number")
    catalyst_flag = cjhif["catalyst_SMILES"].parallel_apply(lambda x: column_check(x))

    # Remove rows where text2smiles translation is faulty
    # Don't consider solvent in this
    filt = reagent_flag & catalyst_flag

    cjhif = cjhif[filt]

    # drop rows where reactants, catalyst and solvent columns are empty
    cjhif_no_empties = cjhif[
        (cjhif["reagent"] != "empty")
        | (cjhif["catalyst"] != "empty")
        | (cjhif["solvent"] != "empty")
    ]

    if logger:
        logger.log(
            {
                f"faulty rows text2smiles: {name}": (~filt).mean(),
                f"rows after additives preprocessing: {name}": len(cjhif),
                f"rows after total empties drop: {name}": len(cjhif_no_empties),
            }
        )

    return cjhif_no_empties


def get_full_reaction_smiles(df, name="cjhif", logger=False):
    """Get full reaction SMILES from the preprocessed dataset by joining
    reagents and catalysts to the original reaction SMILES"""

    print("Generating full reaction smiles (including additives)")

    df["full_reaction_smiles"] = df.parallel_apply(rxn_utils.join_additives, axis=1)

    if logger:
        logger.log({f"rows after join additives: {name}": len(df)})
    return df


def create_atom_dict(mols):
    """Create a dictionary with the number of atoms of each type in a molecule

    Args:
        mols: list of rdkit.Chem.rdchem.Mol objects

    Returns:
            atom_dict: dict, dictionary with the number of atoms of each type in the molecule

    """
    atom_dict = {}

    for mol in mols:
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_dict[atom_symbol] = atom_dict.get(atom_symbol, 0) + 1

    return atom_dict


def alchemic_filter(rxn_smiles):
    """Flag reaction SMILES if products contain atoms that are not in the reactants

    Args:
        rxn_smiles: str, reaction SMILES

    Returns:
        flag: bool, True if reaction is flagged, False otherwise
    """

    # Parse the reaction SMILES into a ChemicalReaction object
    reaction = ReactionFromSmarts(rxn_smiles)

    # Get reactants and products
    reactants = reaction.GetReactants()
    products = reaction.GetProducts()

    # Create atom count dictionaries for reactants and products
    reactant_atom_dict = create_atom_dict(reactants)
    product_atom_dict = create_atom_dict(products)

    # Check for discrepancies
    for atom, count in product_atom_dict.items():
        if atom not in reactant_atom_dict or count > reactant_atom_dict[atom]:
            return True  # Found an atom in products that's not in reactants or its count is higher

    return False  # No discrepancies found


def canonicalize_filter_reaction(df, column, name="cjhif", by_yield=False, logger=False):
    """
    Canonicalize reaction smiles, drop invalid SMILES, filter duplicated SMILES
    (take SMILES with highest yield).

    Args:
        -df: pd.DataFrame, dataframe containing reactions
        -column: str, name of the column containing rxn SMILES
        -by_yield: bool, filter duplicate reactions by yield, keeping the reaction with max yield

    Out:
        -filtered_df: pd.DataFrame, df with an extra column for canonical reaction SMILES and no duplicates
    """

    print("Generating canonical reaction smiles (including additives)")

    # Canonicalize reaction SMILES, create new column for that
    df["canonic_rxn"] = df[column].parallel_apply(lambda x: rxn_utils.canonical_rxn(x))

    # Drop invalid SMILES
    df = df[df["canonic_rxn"] != "Invalid SMILES"]

    if by_yield:
        # take repeated reaction with highest yield
        high_duplicates = df.iloc[
            df[df.duplicated(subset=["canonic_rxn"], keep=False)]
            .reset_index(drop=False)
            .groupby("canonic_rxn")["yield"]
            .idxmax()
            .values
        ]

        # create clean df (no duplicated SMILES)
        filtered_df = pd.concat([df.drop_duplicates("canonic_rxn", keep=False), high_duplicates])

    else:
        filtered_df = df.drop_duplicates("canonic_rxn")

    # last, apply alchemic filter to delete reactions with products with atoms that are not in the reactants
    print("Applying alchemic filter")
    filtered_df_alchemic = filtered_df[~filtered_df["canonic_rxn"].parallel_apply(alchemic_filter)]

    # drop full_reaction_smiles column
    filtered_df_alchemic = filtered_df_alchemic.drop(columns=["full_reaction_smiles"])

    if logger:
        logger.log(
            {
                f"rows after canonicalization: {name}": len(df),
                f"rows after filter duplicates by yield: {name}": len(filtered_df),
                f"rows after alchemic filter: {name}": len(filtered_df_alchemic),
            }
        )

    return filtered_df_alchemic


def clean_USPTO(df, logger=False):
    """Create canonical SMILES column for USPTO and clean it using the same function
    that was applied to out dataset
    """
    print("Cleaning USPTO")

    def _canonical_rxn(rxn_smi):
        try:
            # set reaction type to use fragment bonds (~)
            rxn_type = ReactionFormat.STANDARD_WITH_TILDE

            # parse full reaction SMILES
            ReactEq = parse_extended_reaction_smiles(rxn_smi, remove_atom_maps=True)

            # If no reactants or no products, return invalid smiles
            if len(ReactEq.reactants) * len(ReactEq.products) == 0:
                return "Invalid SMILES"

            # Standard reaction: canonicalize reaction and sort compounds
            std_rxn = sort_compounds(canonicalize_compounds(merge_reactants_and_agents(ReactEq)))

            # Create final reaction SMILES
            rxn = to_reaction_smiles(std_rxn, rxn_type)

            return rxn
        except:
            return "Invalid SMILES"

    def _main_reactant(smiles):
        """Auxiliary function to find main reactant from USPTO set"""

        reactants = parse_extended_reaction_smiles(smiles).reactants

        # remove non-atom characters
        simple_reacts = [re.sub(r"[\[\]\+\-.=()#~*@]", "", i) for i in reactants]

        # Take index of longest SMILES as a rough estimation of main reactant
        max_idx = np.argmax(np.array([len(i) for i in simple_reacts]))

        return reactants[max_idx]

    print("Canonicalizing USPTO")
    df["canonic_rxn"] = df["full_reaction_smiles"].progress_apply(lambda x: _canonical_rxn(x))
    df = df[df["canonic_rxn"] != "Invalid SMILES"].reset_index(drop=True)
    print("Estimating main reactants in USPTO")
    df["main_reactant"] = df["full_reaction_smiles"].progress_apply(lambda x: _main_reactant(x))

    filtered_df = df.drop_duplicates("canonic_rxn")

    if logger:
        logger.log(
            {
                f"rows after canonicalization: uspto": len(df),
                f"rows after filter by yield: uspto": len(filtered_df),
            }
        )
    return filtered_df


if __name__ == "__main__":
    print(full_dict["tetrabutyl ammonium fluoride"])
    cjhif = preprocess_additives("data/raw/", "data_from_CJHIF_utf8")
    cjhif.to_csv("data/cjhif_filtered.tsv", sep="\t")
    canon = canonicalize_filter_reaction(cjhif, "rxn_smiles", by_yield=True)
    canon.to_csv("data/cjhif_canonical.tsv", sep="\t")
