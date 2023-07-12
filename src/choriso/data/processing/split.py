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


def data_split_random(data_path, out_folder, test_frac=0.1, val_frac=0.1, replace_tilde=True):
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

    columns = ["canonic_rxn", "rxnmapper_aam", "rxnmapper_confidence", "yield"]

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
    new_atoms_order = atoms[rotation_index % len(atoms) :] + atoms[: rotation_index % len(atoms)]
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
    data_path, out_folder, test_frac=0.1, val_frac=0.1, replace_tilde=True, augment=False
):
    """Function to split data for reaction forward prediction based on products.

    Args:
        data_path: str, path to clean dataset
        out_folder: str, folder to save the data
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

    # Create products column
    df["products"] = df["canonic_rxn"].apply(lambda x: x.split(">>")[1])

    # Split dataset into train, validation and test based on reaction products
    print("Splitting data")
    train, test = dataset_product_split(df, test_frac)

    train, val = dataset_product_split(train, val_frac)

    shuffled_train = train.sample(frac=1.0, random_state=42)

    if replace_tilde:
        shuffled_train = shuffled_train.replace("~", ".", regex=True)
        val = val.replace("~", ".", regex=True)
        test = test.replace("~", ".", regex=True)

    columns = ["canonic_rxn", "rxnmapper_aam", "rxnmapper_confidence", "yield"]

    if "template_r0" in df.columns:
        columns = columns + ["template_r0", "template_r1"]

    # Save the datasets but keep only some columns
    shuffled_train = shuffled_train[columns]

    if augment:
        print("Augmenting SMILES...")
        # create a copy of train df
        train_aug = shuffled_train.copy()
        # rotate reactants
        train_aug["canonic_rxn"] = train_aug["canonic_rxn"].apply(lambda x: rotate_rxn(x))
        # mix original and rotated rxns
        shuffled_train = pd.concat([shuffled_train, train_aug], ignore_index=True)
        # shuffle
        shuffled_train = shuffled_train.sample(frac=1.0, random_state=33)

    test = test[columns]
    val = val[columns]
    df = df[columns]

    # Save splits
    products_path = out_folder + "products_split/"
    if not os.path.isdir(products_path):
        os.mkdir(products_path)

    shuffled_train.to_csv(products_path + "choriso_products_train.tsv", sep="\t")
    test.to_csv(products_path + "choriso_products_test.tsv", sep="\t")
    val.to_csv(products_path + "choriso_products_val.tsv", sep="\t")


def data_split_mw(data_path, low_mw=150, high_mw=700):
    """Function to split data for reaction forward prediction based on molar weight.

    Args:
        data_path: str, path to clean dataset
        high_mw: float, molar weight to split test and train sets
        mode: str, 'high' or 'low', if 'high' the test set will contain reactions with
                molar weight > mw, if 'low' the test set will contain reactions with
                molar weight < mw
    Out:
        train, val, test: tuple, pd.Dfs containing train, validation and test data
    """

    # Read dataset
    print("Reading dataset")

    df = pd.read_csv(data_path + "choriso.tsv", sep="\t").fillna(0)

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
        columns = ["canonic_rxn", "rxnmapper_aam", "rxnmapper_confidence", "yield"]
        if "template_r0" in df.columns:
            columns = columns + ["template_r0", "template_r1"]

        train = train[columns]
        test = test[columns]
        val = val[columns]

        return train, test, val

    # High MW
    choriso_test_high = df[df["MolWt"] >= high_mw]
    choriso_train_high = df[df["MolWt"] < high_mw]

    train, val, test = _get_split(choriso_train_high, choriso_test_high)
    assert len(train) + len(val) + len(test) == len(df)

    # Save splits
    highmw_path = data_path + "high_mw_split/"
    if not os.path.isdir(highmw_path):
        os.mkdir(highmw_path)

    train.to_csv(highmw_path + "choriso_high_train.tsv", sep="\t")
    test.to_csv(highmw_path + "choriso_high_test.tsv", sep="\t")
    val.to_csv(highmw_path + "choriso_high_val.tsv", sep="\t")

    # Low MW
    choriso_test_low = df[df["MolWt"] <= low_mw]
    choriso_train_low = df[df["MolWt"] > low_mw]

    train, val, test = _get_split(choriso_train_low, choriso_test_low)
    assert len(train) + len(val) + len(test) == len(df)

    # Save splits
    lowmw_path = data_path + "low_mw_split/"
    if not os.path.isdir(lowmw_path):
        os.mkdir(lowmw_path)

    train.to_csv(lowmw_path + "choriso_low_train.tsv", sep="\t")
    test.to_csv(lowmw_path + "choriso_low_test.tsv", sep="\t")
    val.to_csv(lowmw_path + "choriso_low_val.tsv", sep="\t")
