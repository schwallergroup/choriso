"""Main preprocessing pipeline"""

import os
import time

import click

import wandb
from choriso.data import *


def df_download_step(data_dir, uspto=False):
    """Download raw dataset"""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.isfile(data_dir + "raw_dataset.csv"):
        print("Downloading raw CJHIF")
        preproc.download_raw_data(data_dir)

    if uspto:
        download_full_USPTO(data_dir)


def df_cleaning_step(data_dir, raw_file_name, out_dir, name, logger):
    """Execute dataset cleaning step.

    Process additives: text2smiles w leadmine
    Combine reagents
    Reaction smiles canonicalization

    Args:
        data_dir (str): Directory where the raw data is stored.
        raw_file_name (str): Name of the raw file.
        out_dir (str): Directory where the processed data will be stored.
        name (str): Name of the dataset.
        logger (Logger): Logger object.

    """

    if name == "cjhif":
        # Get SMILES from text using leadmine
        df = preproc.preprocess_additives(
            data_dir, raw_file_name, "cjhif", logger
        )

        # Create full reaction SMILES
        df = preproc.get_full_reaction_smiles(df, "cjhif", logger)

        # Canonicalize and filter
        df = preproc.canonicalize_filter_reaction(
            df,
            "full_reaction_smiles",
            name,
            name == "cjhif",  # Only clean by yield for cjhif
            logger,
        )

    if name == "uspto":
        # If loading uspto, don't have to process additives
        df = pd.read_csv(
            data_dir + raw_file_name,
            sep="\t",
            usecols=["ReactionSmiles", "CalculatedYield"],
        ).rename(
            columns={
                "ReactionSmiles": "full_reaction_smiles",
                "CalculatedYield": "yield",
            }
        )

        df = clean_USPTO(df, logger)

    # Save processed dataset
    df.to_csv(out_dir + f"{name}_processed_clean.tsv", sep="\t", index=False)

    print(f"\nFinished cleaning step for {name}.\n")


def df_atom_map_step(
    out_dir,
    name,
    logger,
    batch_size,
    namerxn=False,
    testing=False,
):
    """Atom mapping based calculations.

    Calculate aam for processed reactions using
    - RXNMapper
    - NameRXN
    Compare results by comparing active entities in the reaction, as classified using the different aam methods.

    Args:
        out_dir (str): Directory where the processed data is stored.
        name (str): Name of the dataset.
        logger (Logger): Logger object.
        batch_size (int): Batch size for NameRXN.
        namerxn (bool, optional): Whether to calculate NameRXN aam. Defaults to False.
        testing (bool, optional): Testing code. Use testing parameters for batching (smaller set). Defaults to False.

    """
    filepath = out_dir + f"{name}_processed_clean.tsv"

    if not os.path.isfile(filepath):
        raise Exception(
            "Atom mapping step couldn't be started. Make sure the cleaning step finished successfully."
        )

    df = pd.read_csv(
        out_dir + f"{name}_processed_clean.tsv",
        sep="\t",
    )

    if testing:
        timeout = 30
        batch_sz = 10
        tmp_dir = "data/test/tmp/"
    else:
        timeout = 500
        batch_sz = 300
        tmp_dir = "data/tmp/"

    ### Remove reactions that can't be atom mapped with either tool
    print("Cleaning dataset previous to atom mapping.")
    df = atom_map.clean_preproc_df(df, "canonic_rxn", name, logger)

    # Calculate first RXNMapper
    print(f"Calculating RXNMapper atom mapping for {name}")
    df = atom_map.atom_map_rxnmapper(df, "canonic_rxn", name, logger)

    if namerxn:
        # Calculate namerxn aam
        print(f"Calculating NameRXN atom mapping for {name}")
        mapped_smi = atom_map.atom_map_hazelnut(
            df,
            "canonic_rxn",
            timeout=timeout,
            batch_sz=batch_sz,
            tmp_dir=tmp_dir,
            logger=logger,
        )

        if type(mapped_smi) != int:
            df = pd.concat([df, mapped_smi], axis=1)

        print(f"\nFinished calculating atom mappings for {name}.\n")

        ######### Handling inconsistencies in aams

        n_bad_format = (mapped_smi["can_canon"] == 0).sum()
        logger.log({f"rxns NameRXN bad formatting: {name}": n_bad_format})

        # Save this subset for posterior analysis
        (
            df.query("can_canon==0")
            .loc[:, ["nm_aam", "canonic_rxn"]]
            .to_csv(out_dir + "errors_nmrxn.tsv", index=False, sep="\t")
        )

        # Change these by nan, so they're replaced by RXNMapper aam
        df["nm_aam"].replace("pailas socio", np.nan, inplace=True)

        # Fill timeouts in nm_aam with values from rxnmapper
        df["nm_aam"].fillna(df["rxnmapper_aam"], inplace=True)

        ################## Finish filling inconsistencies

        # Comparing aams through reagent classification
        if name == "cjhif":
            rgt_set_rxnmapper = (
                df["rxnmapper_aam"].apply(aam_reagent_classify).astype(str)
            )
            rgt_set_namerxn = (
                df["nm_aam"].apply(aam_reagent_classify).astype(str)
            )

            # Flag rows where active entities match.
            df["aam_matches"] = rgt_set_rxnmapper == rgt_set_namerxn
            logger.log(
                {
                    f"aam agreement cjhif abs:": df["aam_matches"].sum(),
                    f"aam agreement cjhif percent:": df["aam_matches"].sum()
                    / df.shape[0],
                }
            )

            # Drop redundant/no longer useful columns
            # df.drop(columns=["reagent", "catalyst", "full_reaction_smiles"], inplace=True)

            # Save clean version of dataset (choriso)
            choriso = df.query("aam_matches").drop(
                columns=["nm_aam", "aam_matches"]
            )

            choriso.to_csv(out_dir + f"choriso.tsv", sep="\t", index=False)

            logger.log({"final dataset size": choriso.shape[0]})

    # Save original version of dataset too
    # Works for both cjhif and uspto
    df.to_csv(
        out_dir + f"{name}_atom_mapped_dataset.tsv", sep="\t", index=False
    )


def df_stereo_check_step(data_dir, name, logger):
    """Check stereochemistry of reactions and remove wrong ones. We can add
    additional checks here if needed.

    Args:
        data_dir (str): Directory where the processed data is stored.
        name (str): Name of the dataset.
        logger (Logger): Logger object.

    """

    filepath = data_dir + f"{name}_atom_mapped_dataset.tsv"

    # open file
    df = pd.read_csv(
        filepath,
        sep="\t",
    )

    # flag reactions with stereo issues
    df["stereo_wrong"] = df["canonic_rxn"].parallel_apply(
        stereo.flag_stereoalchemy
    )

    # correct stereo issues (we simply remove the stereocenters so we keep the reaction)
    df["canonic_rxn"] = df.parallel_apply(
        lambda x: stereo.remove_chiral_centers(x["canonic_rxn"])
        if x["stereo_wrong"] == True
        else x["canonic_rxn"],
        axis=1,
    )

    # correct mapping for reactions with stereo issues
    df["rxnmapper_aam"] = df.parallel_apply(
        lambda x: stereo.remove_chiral_centers(x["rxnmapper_aam"])
        if x["stereo_wrong"] == True
        else x["rxnmapper_aam"],
        axis=1,
    )

    # remove reactions where products do not contain carbon, this is mainly due to an issue with USPTO
    df["has_carbon"] = df["canonic_rxn"].parallel_apply(
        lambda x: stereo.has_carbon(x.split(">>")[1])
    )

    df = df[df["has_carbon"] == True]

    # drop duplicates
    df = df.drop_duplicates(subset=["canonic_rxn"])
    

    if name == "uspto":
        # remove this specific reaction (it's giving problems when featurizing for G2S)
        df = df[~df["canonic_rxn"].str.contains("C.O>>C.O.O.O.O.O")]

    print("Saving final dataset.")
    if name == "cjhif":
        name = "choriso"

    # save the final dataset and a public version which we will share (without info from nameRXN)
    df.to_csv(data_dir + f"{name}.tsv", sep="\t", index=False)

    if name == "uspto":
        df_public = df[["canonic_rxn", "rxnmapper_aam", "yield"]]

    else:
        # pick only canonical_rxn and rxnmapper_amm columns
        df_public = df[
            [
                "canonic_rxn",
                "rxnmapper_aam",
                "reagent",
                "solvent",
                "catalyst",
                "yield",
            ]
        ]

    # save public version
    df_public.to_csv(data_dir + f"{name}_public.tsv", sep="\t", index=False)


def df_splitting_step(
    data_dir, out_dir, file_name, low_mw, high_mw, augment, templates
):
    """Split the data into multiple splits. We get training and validation splits by product,
    and test splits by product, MW and random. Optionally you can also get an augemented
    training set.

    Args:
        data_dir (str): path to data directory
        out_dir (str): path to output directory
        file_name (str): name of the file to split
        low_mw (float): lower bound of MW for splitting by MW
        high_mw (float): upper bound of MW for splitting by MW
        augment (bool): whether to augment SMILES for the product split
        templates (bool): whether to compute templates on the test set

    """
    # path to clean df
    path = data_dir + file_name

    # Split data by products to train, val, test sets
    processing.split.data_split_by_prod(
        path,
        out_dir,
        file_name,
        low_mw=low_mw,
        high_mw=high_mw,
        test_frac=0.07,
        val_frac=0.077,
        augment=augment,
        templates=templates,
    )


@click.command()
@click.option("--data-dir", type=click.Path(), default="data/raw/")
@click.option("--report-dir", type=click.Path(), default="data/report/")
@click.option("-o", "--out-dir", type=click.Path(), default="data/processed/")
@click.option("--download_raw", is_flag=True)
@click.option("--download_processed", is_flag=True)
@click.option(
    "--run",
    "-r",
    type=click.Choice(["clean", "atom_map", "stereo", "split"]),
    multiple=True,
)
@click.option(
    "--wandb_log", is_flag=True, help="Log results using Weights and Biases."
)
@click.option(
    "--uspto",
    is_flag=True,
    help="Run preprocessing also on the USPTO full dataset.",
)
@click.option("--batch", default=200, help="Batch size for rxnmapper")
@click.option(
    "--namerxn",
    is_flag=True,
    help="Run atom mapping with Namerxn and comparison with RXNMapper.",
)
@click.option(
    "--testing",
    is_flag=True,
    help="Testing code. Use testing parameters for batching (smaller set)",
)
@click.option("--split_file_name", default="choriso.tsv")
@click.option(
    "--templates",
    is_flag=True,
    help="Compute templates on the test set",
)
@click.option(
    "--augment",
    is_flag=True,
    help="Augment SMILES by creating one additional random SMILES for each product",
)
@click.option(
    "--low_mw",
    default=150,
    help="Lower MW threshold for dataset splitting by MW",
)
@click.option(
    "--high_mw",
    default=700,
    help="Higher MW threshold for dataset splitting by MW",
)
def main(
    data_dir,
    report_dir,
    out_dir,
    download_raw,
    download_processed,
    run,
    wandb_log,
    uspto,
    batch,
    namerxn,
    testing,
    split_file_name,
    low_mw,
    high_mw,
    augment,
    templates,
):
    """Main data preprocessing pipeline."""

    # Setup logger
    if testing:
        logger = Logger(wandb_log, wandb_project="cjhif-test-log")
    else:
        logger = Logger(wandb_log)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # Download datasets
    if download_raw:
        df_download_step(data_dir, uspto)

    if download_processed:
        download_processed_data(out_dir)

    # Start cleaning/preprocessing
    if "clean" in run:
        if not os.path.exists(out_dir + "cjhif_processed_clean.tsv"):
            df_cleaning_step(
                data_dir, "data_from_CJHIF_utf8", out_dir, "cjhif", logger
            )
        if uspto:
            df_cleaning_step(
                data_dir, "merged_USPTO.rsmi", out_dir, "uspto", logger
            )

    if "atom_map" in run:
        if not os.path.exists(out_dir + "cjhif_atom_mapped_dataset.tsv"):
            df_atom_map_step(out_dir, "cjhif", logger, batch, namerxn, testing)

        if uspto:
            df_atom_map_step(out_dir, "uspto", logger, batch, namerxn, testing)

        print("Finished atom mapping step.")

    if "stereo" in run:
        print("Starting stereo check step.")
        # Add stereo processing to get final dataset
        df_stereo_check_step(out_dir, "cjhif", logger)

        if uspto:
            print("Starting stereo check step for USPTO.")
            df_stereo_check_step(out_dir, "uspto", logger)

    if "split" in run:
        df_splitting_step(
            out_dir,
            out_dir,
            split_file_name,
            low_mw,
            high_mw,
            augment,
            templates,
        )


if __name__ == "__main__":
    main()
