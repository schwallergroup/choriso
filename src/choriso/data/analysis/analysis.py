"""Define functions for data analysis"""

import os
import tarfile

import matplotlib.pyplot as plt
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.AllChem import ReactionFromSmarts
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters, CalcNumRings
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rxn.chemutils.miscellaneous import is_valid_smiles
from tqdm.auto import tqdm

tqdm.pandas()


def download_full_USPTO(data_dir):
    """Download full USPTO.
    This file contains the merged reactions from USPTO patents
    and applications.
    """

    url = "https://drive.switch.ch/index.php/s/TkZc0GoLKvFuBq2/download"
    target_path = data_dir + "full_USPTO.tar.gz"

    if not os.path.isfile(target_path):
        print("Downloading full USPTO...")

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())

        # Decompress tar file
        with tarfile.open(data_dir + "full_USPTO.tar.gz") as f:
            f.extractall(data_dir)


def delta_stereo(df, reactant_mols):
    """Compute ring difference between products and reactants.

    Args:
       -df: pd.DataFrame, df containing properties
       -reactant_mols: pd.Series, mol objects from reactants

    Out:
       -delta: pd.Series, stereocenters difference between products and reactants
    """
    react_stereocenters = reactant_mols.apply(CalcNumAtomStereoCenters)
    delta = df["stereocenters"] - react_stereocenters

    return delta


def delta_rings(df, reactant_mols):
    """Compute ring difference between reagents and products

    Args:
       -df: pd.DataFrame, df containing properties
       -reactant_mols: pd.Series, mol objects from reactants

    Out:
       -delta: pd.Series, ring difference between products and reactants
    """
    react_rings = reactant_mols.apply(CalcNumRings)
    delta = df["RingCount"] - react_rings

    return delta


def compute_properties(
    df_orig,
    rxn_col,
    descriptors=[
        "TPSA",
        "MolLogP",
        "MolWt",
        "NumHAcceptors",
        "NumHDonors",
        "RingCount",
        "NumAromaticHeterocycles",
    ],
    in_precursors=["Pd", "Al", "Li", "Mg"],
):
    """Take a df containing canonicalized (valid) rxn SMILES and create a df with desired properties.

    Args:
        -df_orig: pd.DataFrame, df containing reactions
        -descriptors: list, rdkit descriptors to compute
        -in_precursors: list, chemical species to check in reagents
    Out:
        -df_properties: pd.DataFrame, df containing computed properties for a given dataset
    """
    df = df_orig[["canonic_rxn", "rxnmapper_aam", "rxnmapper_confidence", "rxn_class"]].copy()

    # Add a column with reaction products and number of products per reaction
    df["products"] = df["canonic_rxn"].apply(lambda x: x.split(">>")[1])
    df["n_products"] = df["products"].apply(lambda x: len(x.split(".")))

    # Add reagents/reactants
    df["precursors"] = df["canonic_rxn"].apply(lambda x: x.split(">>")[0])

    # Add main reactant (we use original rxn_smiles)
    if not "main_reactants" in df_orig.columns:
        df["main_reactant"] = df_orig[rxn_col].apply(lambda x: x.split(">>")[0])
    else:
        df["main_reactant"] = df_orig["main_reactant"]

    # Convert products into rdkit objects
    prod_mols = df["products"][df["products"].apply(is_valid_smiles)].apply(Chem.MolFromSmiles)

    # Convert main reactant into rdkit object
    react_mols = df["main_reactant"][df["main_reactant"].apply(is_valid_smiles)].apply(
        Chem.MolFromSmiles
    )

    # Rdkit descriptors
    calculator = MolecularDescriptorCalculator(descriptors)

    # Compute descriptors (for reaction product)
    properties = prod_mols.apply(calculator.CalcDescriptors)

    # Df with one column for each descriptor
    computed_df = pd.DataFrame(properties.tolist(), columns=descriptors, index=properties.index)

    # Append properties df
    df = pd.concat((df, computed_df), axis=1)

    # Calculate product stereocenters
    df["stereocenters"] = prod_mols.apply(CalcNumAtomStereoCenters)

    # Calculate stereocenters difference
    df["delta_stereocenters"] = delta_stereo(df, react_mols)

    # Calculate ring difference
    df["delta_rings"] = delta_rings(df, react_mols)

    # Compute if precursors contain any of the specified SMILES
    for smi in in_precursors:
        df["has_" + smi] = df["precursors"].apply(lambda x: smi in x)

    return df


def distrib_reaction_types(rxn_name_list, name):
    """
    Calculate the distributions of reaction types, as given by NameRXN.

    Count 3 levels (a.b.c)
    """

    def _count_level_i(v, i=1):
        lvl_freq = (
            v.astype(str).apply(lambda x: ".".join(x.split(".")[:i])).rename(name).value_counts()
        )
        return lvl_freq

    rxn_name_list.replace("0.0", "0.0.0", inplace=True)

    lvl1 = _count_level_i(rxn_name_list, 1)
    lvl2 = _count_level_i(rxn_name_list, 2)
    lvl3 = _count_level_i(rxn_name_list, 3)

    return (lvl1, lvl2, lvl3)


def n_reacting_atoms(
    df,
    name,
):
    """
    Count the number of reacting atoms for each reaction in a dataset.
    Args:
        df: pd.DataFrame with a 'rxnmapper_aam' field containing mapped reactions
        name: name of the dataset
    """
    from itertools import chain

    def _n_from_rxn(smi):
        """Count number of reacting atoms in a single reaction"""
        try:
            rxn = AllChem.ReactionFromSmarts(smi, useSmiles=True)
            rxn.Initialize()
            rxn.RemoveUnmappedReactantTemplates()
            reacting_atoms = rxn.GetReactingAtoms()
            return len(list(chain(*reacting_atoms)))
        except:
            # TODO check these reactions, are problematic for rdkit
            return 30

    return df["rxnmapper_aam"].apply(_n_from_rxn).rename(name).value_counts()


def log_dataset_info(df, orig_size, name, logger):
    """Calculate basic information about dataset and log it.

    Args:
       df: pd.Dataframe obtained from compute_properties()
       name: str, dataset name
       out: output file path
    """

    logger.log("*" * 100 + "\n")
    logger.log("\n" + name + "\n")
    logger.log("\nOriginal size: {0}".format(orig_size))
    logger.log("\nSize after canonicalization and duplicates dropping: {0}\n".format(len(df)))

    # Products information
    logger.log("\nPRODUCTS\n")
    distrib_prods = (
        df["n_products"]
        .value_counts()
        .reset_index()
        .rename(columns={"n_products": "num_reactions", "index": "num_products_per_reaction"})
    )
    distrib_prods = distrib_prods.set_index("num_products_per_reaction")
    distrib_prods["relative"] = (
        distrib_prods["num_reactions"] * 100 / distrib_prods["num_reactions"].sum()
    )
    logger.log(str(distrib_prods.head(10)))

    # Number of different products
    logger.log("\n\nNumber of different products: {0}\n".format(len(set(df["products"]))))
    # Count reactions per product
    count = df.groupby("products").size().sort_values(ascending=False).rename("Count clean rxns")

    logger.log(f"\n\nProducts with more than 1 reaction: {(count>1).sum()}\n\n")
    logger.log(str(count.head()))
    logger.log("\n\n")

    # Stereochemistry information
    logger.log("\n\nSTEREOCHEMISTRY\n")
    stereos = len(df[df["stereocenters"] != 0])
    logger.log(
        "\nNumber of reactions with stereocenters: {0} ({1}% of the total)".format(
            stereos, stereos / len(df) * 100
        )
    )
    logger.log("\n\n")


def log_reactions_info(df, logger, n_rxns=100, seed=42):
    """Take random reactions from dataset and show the canonical
    reaction and the atom-mapped reaction as an image.
    """

    for idx, row in df.sample(n_rxns, random_state=seed).iterrows():
        # Draw rxn
        rxn = ReactionFromSmarts(row["canonic_rxn"], useSmiles=True)
        img = Draw.ReactionToImage(rxn, subImgSize=(300, 300))

        # Draw mapped rxn
        rxn_aam = ReactionFromSmarts(row["rxnmapper_aam"], useSmiles=True)
        img_aam = Draw.ReactionToImage(rxn_aam, subImgSize=(300, 300))

        # Build fig
        w, h = img.width / 30, img.height / 30
        fig, ax = plt.subplots(2, 1, figsize=(w, h * 2))
        ax[0].imshow(img)
        ax[1].imshow(img_aam)

        ax[0].set_title(f"Test idx: {idx}, NameRXN: {row['rxn_class']}", fontsize=50)

        ax[0].axis("off")
        ax[1].axis("off")

        logger.log_fig(fig, "Test reaction samples.")
