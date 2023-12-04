"""This module contains the code for selectivity metrics"""

import signal

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction
from transformers import logging

pandarallel.initialize(progress_bar=True, nb_workers=22)

logging.set_verbosity_error()  # Only log errors


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


def template_smarts_from_mapped_smiles(
    mapped_smiles,
    radius=1,
    failed_template=False,
    return_raw_template=False,
):  
    """Extract reaction template from mapped reaction SMILES.

    Args:
        mapped_smiles: str, mapped reaction SMILES
        radius: int, radius of the reaction template
        failed_template: bool, if True, return the list of reactions for which the template could not be extracted
        return_raw_template: bool, if True, return the raw template (canonical)

    Out:
        template: str, reaction template
    """
    
    if type(mapped_smiles) == str:
        rxn = ChemicalReaction(mapped_smiles, clean_smiles=False)
        failed_templates = []
        try:
            rxn.generate_reaction_template(radius)
            if return_raw_template:
                return rxn.canonical_template
            else:
                return rxn.canonical_template.smarts
        except:
            if failed_template != False:
                failed_templates.append(mapped_smiles)

    if failed_template != False:
        print(
            "Problem generating the templates of the following reactions: \n",
            failed_templates,
        )

    return ""


def co2_transform(absval, mode="co2"):
    """Transform absolute CO2 and kWh into relative scale.

    Args:
        absval: float, CO2 or kWh value
        mode: str, 'co2' or 'time', default 'co2'
    """

    if mode == "co2":
        k = 5e-1
    elif mode == "time":
        k = 5e-3

    y = 100 * np.exp(-k * absval)

    return y


def top_n_accuracy(df, n):
    """Take predicition dataframe and compute top-n accuracy

    Args:
        df (pd.DataFrame): dataframe with predictions
        n (int): number of top predictions to consider

    Returns:
        float: top-n accuracy
    """

    correct = 0

    for i, row in df.iterrows():
        for i in range(n):
            if row["target"] == row[f"pred_{i}"]:
                correct += 1
                break

    accuracy = round(correct / len(df) * 100, 1)

    return accuracy


def flag_regio_problem(rxn, *args):
    """Flag regioselectivity problems. For the moment only one-product
    reactions. The function extracts the reaction template (only reacting atoms) and then checks
    if the matching atoms in the product can generate several products.

    Args:
        rxn: str, reaction SMILES
        *args: tuple, optional reaction mapping and template with radius=1.
                If not provided, they are computed.

    Out:
        bool, True if the reaction is regioselective, False otherwise
    """

    if len(args) == 1:
        map_rxn, template = args[0][0], args[0][1]
        if template == False:
            return False

    elif len(args) == 0:
        # extract rxn template
        map_rxn = aam_from_smiles([rxn])[0]["mapped_rxn"]
        template = template_smarts_from_mapped_smiles(map_rxn, radius=1)

    def _sanitize_filter_prods(prods):
        good = []
        for prod in prods:
            try:
                x = Chem.SanitizeMol(prod[0])
                good.append(Chem.MolToSmiles(prod[0]))
            except Chem.MolSanitizeException:
                pass
        return set(good)

    def _check_template(temp):
        try:
            reaction = AllChem.ReactionFromSmarts(temp)
            return True
        except ValueError:
            return False

    # proceed only if template exists
    check = _check_template(template)

    if check:
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


def regio_score(df, negative_acc=False):
    """Regioselectivity classification score. Top1 accuracy for reactions with regioselectivity
    problems.

    Args:
        df: pd.DataFrame, dataframe with predictions and flags
        negative_acc: bool

    Returns:
        acc: float, top-1 accuracy for reactions where regioselectivity is an issue
    """

    if "regio_flag" not in df.columns:
        raise ValueError("regio_flag column not found in dataframe")

    df_true = df[df["regio_flag"] == True]
    df_false = df[df["regio_flag"] == False]

    # check if products are the same
    true_prods = np.array(df_true["target"].values)
    pred_prods = np.array(df_true["pred_0"].values)

    acc = true_prods == pred_prods
    acc = round(np.sum(acc) / len(acc) * 100, 1)

    if negative_acc:
        # check if products are the same
        true_prods = np.array(df_false["target"].values)
        pred_prods = np.array(df_false["pred_0"].values)

        acc_neg = true_prods == pred_prods
        acc_neg = round(np.sum(acc_neg) / len(acc_neg) * 100, 1)

        return acc, acc_neg

    else:
        return acc


def flag_stereo_problem(template, rxn=False):
    """Flag stereoselectivity problems.
    Args:
        template: str, extracted template with radius=0 from reaction SMILES
        rxn: str, reaction SMILES (in case template is not provided)

    Out:
        bool, True if the reaction has stereoselectivity issues, False otherwise
    """

    if rxn:
        # extract rxn template
        map_rxn = aam_from_smiles([rxn])[0]["mapped_rxn"]
        template = template_smarts_from_mapped_smiles(map_rxn)

    try:
        temp_prods = template.split(">>")[1].split(".")
        # check if any of the strings in prods contain '@'
        if any("@" in prod for prod in temp_prods):
            return True
        else:
            return False

    except AttributeError:
        return False


def stereo_score(df, negative_acc=False):
    """Stereoselectivity classification score. Top1 accuracy for reactions with stereoselectivity
    problems.

    Args:
        df: pd.DataFrame, dataframe with predictions

    Returns:
        acc: float, top-1 accuracy for reactions where stereoselectivity is an issue
        negative_acc: float, top-1 accuracy for reactions where stereoselectivity is not an issue
    """

    if "stereo_flag" not in df.columns:
        raise ValueError("stereo_flag column not found in dataframe")

    df_true = df[df["stereo_flag"] == True]
    df_false = df[df["stereo_flag"] == False]

    # check if products are the same
    true_prods = np.array(df_true["target"].values)
    pred_prods = np.array(df_true["pred_0"].values)

    acc = true_prods == pred_prods
    acc = round(np.sum(acc) / len(acc) * 100, 1)

    if negative_acc:
        # check if products are the same
        true_prods = np.array(df_false["target"].values)
        pred_prods = np.array(df_false["pred_0"].values)

        acc_neg = true_prods == pred_prods
        acc_neg = round(np.sum(acc_neg) / len(acc_neg), 1)

        return acc, acc_neg

    else:
        return acc

