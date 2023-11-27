"""Functions for operations/features dependent on atom mapping for data preprocessing"""

import os
import re
import signal
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem
from rxnmapper import RXNMapper
from sklearn.utils import gen_batches
from tqdm import tqdm

# Shutdown RXNMapper warnings
from transformers import logging

logging.set_verbosity_error()  # Only log errors
rxnmapper = RXNMapper()

from choriso.data.processing import rxn_utils


def clean_preproc_df(df, col, ds_name, logger):
    """
    Remove reactions that will have trouble with RXNMapper.

    - rxns containing `*`
    - invalid rxns
    - rxns longer than 512 tokens
    """

    # Remove reactions containing "*"
    logger.log("Removing strings with a star")
    query = df[col].str.contains("\*")
    have_star = query.sum()

    df = df[~query]

    # Merge reagents and reactants + canonicalize
    df[col] = df[col].apply(rxn_utils.merge_reagents)

    # Only allow valid reactions
    query = df[col].apply(rxn_utils.is_reaction_valid)
    logger.log("Removing non valid reactions.")
    invalid_smi = sum(~query)
    df[~query][col].to_csv("ti.csv")
    df = df[query]

    # Filter reactions longer than 512 tokens
    suspect = df.loc[df[col].str.len() >= 512]
    logger.log("Removing reactions longer than 512 tokens.")
    query = suspect[
        suspect[col].apply(
            rxn_utils.has_more_than_max_tokens, rxnmapper=rxnmapper
        )
    ].index
    high_token_count = len(query)
    df = df.drop(query)

    logger.log(
        {
            f"rows that contain *: {ds_name}": have_star,
            f"rows that are invalid SMILES: {ds_name}": invalid_smi,
            f"rows that are > 512 tokens: {ds_name}": high_token_count,
        }
    )
    return df


def atom_map_hazelnut(
    df, name, batch_sz=3000, timeout=500, tmp_dir="data/tmp/", logger=False
):
    """
    Calculate atom mapping using HazELNut's atom mapper.
    """

    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    # Store smiles as a file
    unmapped_file = tmp_dir + "to_map_hazelnut.smi"
    mapped_file = tmp_dir + "mapped_hazelnut.smi"

    def _catch_error_rxn(smi):
        """Helper function.

        Catch errors with RDKit-unprocessable reactions after NameRXN aam.
        """
        canon_rxn = rxn_utils.canonical_rxn(smi)
        if canon_rxn == "Invalid SMILES":
            return 0
        return 1

    # Replace "~" by "." in smiles cause NameRXN can't handle this
    df[name] = df[name].str.replace("~", ".")

    # Split df in batches to keep track of progress
    # also allows to early terminate long running pss
    df_batches = np.array_split(df[name], batch_sz)
    mapped_smiles = []

    for i, df_i in enumerate(tqdm(df_batches)):
        try:
            # Remove extended smiles appendix for namerxn
            df_i = df_i.apply(lambda x: re.sub(r"( )+\|f:\d\..*\|", "", x))

            df_i.to_csv(unmapped_file, index=False, header=False)

            # Attempt to calc aam with namerxn
            p = subprocess.Popen(
                ["namerxn", "-complete", unmapped_file, mapped_file],
                # Ignore NameRXN warnings
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Raise timeout if not finished after time limit
            p.wait(timeout=timeout)

            mapped_i = pd.read_csv(mapped_file, sep=" ", header=None)
            mapped_i[2] = mapped_i[0].apply(_catch_error_rxn)

        except subprocess.TimeoutExpired:
            # Kill process
            os.kill(p.pid, signal.SIGTERM)

            # Return a timeout flag for this batch
            mapped_i = pd.DataFrame(
                np.repeat([[np.nan, "0.0", 1]], df_i.shape[0], axis=0)
            )

            # In next step, replace all this timeouts by whatever the RXNMapper outputs.
            if logger:
                logger.log("* Timeout during NameRXN application.")

        mapped_smiles.append(
            mapped_i.rename(
                columns={0: "nm_aam", 1: "rxn_class", 2: "can_canon"}
            )
        )

    # Concat list into single df
    mapped_df = pd.concat(mapped_smiles).reset_index(drop=True)
    # Convert str nan to np.nan so it can be filled after
    mapped_df["nm_aam"].replace("nan", np.nan, inplace=True)
    return mapped_df


def atom_map_rxnmapper(df, col, name, logger, batch_size=200):
    """
    Calculate atom mapping using RXNMapper's atom mapper.
    """

    # Produce atom mappings in batch
    def _atom_map(df, col, batch_size=batch_size):
        map_rxn = []
        for batch in tqdm(
            gen_batches(df.shape[0], batch_size),
            total=df.shape[0] // batch_size + 1,
        ):
            df_slice = df[batch]
            map_rxn += rxnmapper.get_attention_guided_atom_maps(
                df_slice[col], canonicalize_rxns=False
            )
        return pd.DataFrame(map_rxn)

    aam_rxns = _atom_map(df, col).rename(
        columns={
            "mapped_rxn": f"rxnmapper_aam",
            "confidence": f"rxnmapper_confidence",
        }
    )
    # Merge results in dataframe
    new_df = df.reset_index(drop=True).merge(
        aam_rxns, left_index=True, right_index=True, how="inner"
    )

    return new_df


def _is_reagent(x):
    if re.match(r".*\[.+:\d\].*", x):
        return True
    return False


def cleanup_aam(smi):
    """Remove aam numbers of atoms that don't show up in both sides."""

    # Remove grouping elements from smiles
    smi = re.sub(r"\|f:\d\.\d\|", "", smi)
    # Remove artificial atom mapping labels (alabels of atoms that are not mapped in products)
    reac, prod = smi.split(">>")
    for i in re.finditer(r"(?<=[^\*])(:\d+)]", reac):
        if i.group() not in prod:
            reac = re.sub(i.group(), "]", reac)

    return f"{reac}>>{prod}"


def aam_reagent_classify(smi):
    """
    Classify reagents in an atom-mapped smiles string.

    Return a set of molecules that have no atom mapping number
    (i.e. don't contribute with atoms to the products).
    """

    smi = cleanup_aam(smi)
    reactsl = smi.split(">>")[0].split(".")

    # Classify
    return sorted(
        [rxn_utils.canonical_smiles_mol(r) for r in reactsl if _is_reagent(r)]
    )
