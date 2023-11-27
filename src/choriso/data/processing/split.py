"""Data splitting functions"""

import os
import random

import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rxn.chemutils.miscellaneous import is_valid_smiles
from sklearn.model_selection import train_test_split


def dataset_product_split(data, frac):
    """Split reactions dataframe in train and test sets keeping reactions
    with equal products in the same set.

    Args:
        data: pd.Df, dataset containing the reactions
        frac: float, fraction of original data for test set

    Out:
        train, test: tuple, pd.Dfs corresponding to train and test sets

    """

    # Number of reactions in the test set
    tot = round(len(data) * frac)

    # Create groups and shuffle
    groups = data.groupby("products").size().sample(frac=1.0, random_state=42)

    # Create lists of counts and values
    counts = groups.values
    products = groups.index.values

    counter = 0
    test_products = []

    # Append products in list until having a number of reactions > tot
    for i, times in enumerate(counts):
        test_products.append(products[i])
        counter += times
        if counter > tot:
            break

    # Create boolean array to select test reactions
    test_mask = data["products"].isin(test_products)

    test = data[test_mask]
    train = data[~test_mask]

    return train, test


def data_split_random(
    data_path, out_folder, test_frac=0.1, val_frac=0.1, replace_tilde=True
):
    """Do a random split of the choriso dataset.

    Args:
        data_path: str, path to clean dataset
        out_folder: str, folder to save the data
        test_frac: float, fraction of reactions in the test set
        val_frac: float, fraction of reactions in the validation set
        replace_tilde: bool, replace the special character '~' indicating that two chemical
                  species belong to the same molecule with a '.'

    Out:
        train, val, test: tuple, pd.Dfs containing train, validation and test data
    """

    # Read dataset
    print("Reading dataset")
    df = pd.read_csv(data_path, sep="\t")

    # Split dataset into train, validation and test based on reaction products
    print("Splitting data")

    train, test = train_test_split(df, test_size=test_frac, random_state=42)
    train, val = train_test_split(train, test_size=val_frac, random_state=42)

    if replace_tilde:
        train = train.replace("~", ".", regex=True)
        val = val.replace("~", ".", regex=True)
        test = test.replace("~", ".", regex=True)

    columns = ["canonic_rxn", "rxnmapper_aam", "yield"]

    if "template_r0" in df.columns:
        columns = columns + ["template_r0", "template_r1"]
    # Save the datasets but keep only some columns
    train = train[columns]
    test = test[columns]
    val = val[columns]
    df = df[columns]

    # Save splits
    random_path = out_folder + "random_split/"
    if not os.path.isdir(random_path):
        os.mkdir(random_path)

    train.to_csv(random_path + "choriso_random_train.tsv", sep="\t")
    test.to_csv(random_path + "choriso_random_test.tsv", sep="\t")
    val.to_csv(random_path + "choriso_random_val.tsv", sep="\t")


def rotate_smiles(mol):
    """Rotate SMILES by a random number of atoms."""
    n_atoms = mol.GetNumAtoms()
    rotation_index = random.randint(0, n_atoms - 1)
    atoms = list(range(n_atoms))
    new_atoms_order = (
        atoms[rotation_index % len(atoms) :]
        + atoms[: rotation_index % len(atoms)]
    )
    rotated_mol = Chem.RenumberAtoms(mol, new_atoms_order)
    return Chem.MolToSmiles(rotated_mol, canonical=False, isomericSmiles=True)


def rotate_rxn(rxn):
    """Rotate reactants in a reaction SMILES by a random number of atoms."""
    split = rxn.split(">>")
    reactants = split[0].split(".")
    mols = [Chem.MolFromSmiles(smile) for smile in reactants]
    rotated = [rotate_smiles(i) for i in mols]

    rot_reactants = ".".join(rotated)

    rot_rxn = rot_reactants + ">>" + split[1]

    return rot_rxn


def data_split_by_prod(
    data_path,
    out_folder,
    file_name,
    low_mw=150,
    high_mw=700,
    test_frac=0.1,
    val_frac=0.1,
    replace_tilde=True,
    augment=False,
):
    """Function to split data for reaction forward prediction based on products.

    Args:
        data_path: str, path to clean dataset
        out_folder: str, folder to save the data
        file_name: str, name of the file to save
        test_frac: float, fraction of reactions in the test set
        val_frac: float, fraction of reactions in the validation set
        replace_tilde: bool, replace the special character '~' indicating that two chemical
                  species belong to the same molecule with a '.'
        augment: bool, do SMILES augmentation by rotating reactant SMILES and mixing it with original rxns
    Out:
        train, val, test: tuple, pd.Dfs containing train, validation and test data
    """

    # Read dataset
    print("Reading dataset")
    df = pd.read_csv(data_path, sep="\t")

    if replace_tilde:
        df.replace("~", ".", regex=True, inplace=True)

    # columns to keep when saving the splits
    columns = ["canonic_rxn", "rxnmapper_aam", "yield"]

    if "template_r0" in df.columns:
        columns = columns + ["template_r0", "template_r1"]

    # create folder to save splits
    saving_path = out_folder + f'{file_name.split(".")[0]}_splits/'

    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    # Create products column
    df["products"] = df["canonic_rxn"].apply(lambda x: x.split(">>")[1])

    # Calculate MW
    print("Calculating Molecular Weight for all products")
    calc = MolecularDescriptorCalculator(["MolWt"])
    df = df[df["products"].parallel_apply(is_valid_smiles)]
    prod_mols = df["products"].apply(Chem.MolFromSmiles)
    df["MolWt"] = prod_mols.apply(calc.CalcDescriptors).parallel_apply(
        lambda x: x[0]
    )

    print("Splitting by MW")
    high_mw_test = df[df["MolWt"] >= high_mw]
    low_mw_test = df[df["MolWt"] < low_mw]
    medium_mw = df[(df["MolWt"] < high_mw) & (df["MolWt"] >= low_mw)]

    # split by product
    print("Splitting by product")

    remainder_prod, test_prod = dataset_product_split(medium_mw, test_frac)

    # check if products in remainder_prod and test_prod overlap
    assert (
        len(
            set(remainder_prod["products"]).intersection(
                set(test_prod["products"])
            )
        )
        == 0
    )

    train, val = dataset_product_split(remainder_prod, val_frac)

    # save val set
    val[columns].to_csv(
        saving_path + file_name.split(".")[0] + "_prod_val.tsv", sep="\t"
    )

    shuffled_train = train.sample(frac=1.0, random_state=33)

    # finally get random split from the shuffled train set
    final_train, rand = train_test_split(
        shuffled_train, test_size=test_frac, random_state=33
    )

    # save train by product
    final_train[columns].to_csv(
        saving_path + file_name.split(".")[0] + "_prod_train.tsv", sep="\t"
    )

    if augment:
        print("Augmenting SMILES...")
        # create a copy of train df
        train_aug = final_train.copy()
        # rotate reactants
        train_aug["canonic_rxn"] = train_aug["canonic_rxn"].parallel_apply(
            lambda x: rotate_rxn(x)
        )
        # mix original and rotated rxns
        shuffled_train = pd.concat(
            [shuffled_train, train_aug], ignore_index=True
        )
        # shuffle
        shuffled_train = shuffled_train.sample(frac=1.0, random_state=33)
        # save augmented train set
        shuffled_train[columns].to_csv(
            saving_path + file_name.split(".")[0] + "_train_aug.tsv", sep="\t"
        )

    # here create big test file with all the test sets
    # for each split (high, low, prod, random), take the corresponding columns and add an extra column with the split name
    # then concatenate all the splits and save the big test file
    print("Creating big test file...")
    high_test = high_mw_test[columns]
    high_test["split"] = "high"
    low_test = low_mw_test[columns]
    low_test["split"] = "low"
    prod_test = test_prod[columns]
    prod_test["split"] = "prod"
    rand_test = rand[columns]
    rand_test["split"] = "random"

    big_test = pd.concat(
        [high_test, low_test, prod_test, rand_test], ignore_index=False
    )

    big_test.to_csv(
        saving_path + file_name.split(".")[0] + "_all_test.tsv", sep="\t"
    )


def data_split_mw(data_path, file_name, low_mw=150, high_mw=700):
    """Function to split data for reaction forward prediction based on molar weight.

    Args:
        data_path: str, path to clean dataset
        file_name: str, name of the file to save
        low_mw: float, lower bound for molar weight
        high_mw: float, upper bound for molar weight

    """

    # Read dataset
    print("Reading dataset")

    name = file_name.split(".")[0]

    df = pd.read_csv(data_path + name + ".tsv", sep="\t").fillna(0)

    # Create products column
    df["products"] = df["canonic_rxn"].apply(lambda x: x.split(">>")[1])

    df.replace("~", ".", regex=True, inplace=True)

    # Calculate MW
    print("Calculating Molecular Weight for all products")
    calc = MolecularDescriptorCalculator(["MolWt"])
    df = df[df["products"].apply(is_valid_smiles)]
    prod_mols = df["products"].apply(Chem.MolFromSmiles)
    df["MolWt"] = prod_mols.apply(calc.CalcDescriptors).apply(lambda x: x[0])

    # Split dataset into train, validation and test based on molar weight
    print("Splitting data")

    # Get both splits
    def _get_split(train, test):
        train, val = train_test_split(train, test_size=0.06, random_state=42)
        train = train.sample(frac=1.0, random_state=42)  # reshuffle
        columns = [
            "canonic_rxn",
            "rxnmapper_aam",
            "rxnmapper_confidence",
            "yield",
        ]
        if "template_r0" in df.columns:
            columns = columns + ["template_r0", "template_r1"]

        train = train[columns]
        test = test[columns]
        val = val[columns]

        return train, val, test

    # High MW
    choriso_test_high = df[df["MolWt"] >= high_mw]
    choriso_train_high = df[df["MolWt"] < high_mw]

    train, val, test = _get_split(choriso_train_high, choriso_test_high)
    assert len(train) + len(val) + len(test) == len(df)

    # Save splits
    highmw_path = data_path + "high_mw_split/"
    if not os.path.isdir(highmw_path):
        os.mkdir(highmw_path)

    train.to_csv(highmw_path + name + "_high_train.tsv", sep="\t")
    test.to_csv(highmw_path + name + "_high_test.tsv", sep="\t")
    val.to_csv(highmw_path + name + "_high_val.tsv", sep="\t")

    # Low MW
    choriso_test_low = df[df["MolWt"] <= low_mw]
    choriso_train_low = df[df["MolWt"] > low_mw]

    train, val, test = _get_split(choriso_train_low, choriso_test_low)
    assert len(train) + len(val) + len(test) == len(df)

    # Save splits
    lowmw_path = data_path + "low_mw_split/"
    if not os.path.isdir(lowmw_path):
        os.mkdir(lowmw_path)

    train.to_csv(lowmw_path + name + "_low_train.tsv", sep="\t")
    test.to_csv(lowmw_path + name + "_low_test.tsv", sep="\t")
    val.to_csv(lowmw_path + name + "_low_val.tsv", sep="\t")
