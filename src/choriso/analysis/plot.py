#Module with functions to plot stuff
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw

from choriso import analysis


def n_reactions_per_product(
        dfs,
        names,
        logger
):
    """Compute and plot distribution of number of reactions per product."""

    fig, ax = plt.subplots(1, len(dfs),
                           figsize=(15,4),
                           sharey=True)

    for i, df in enumerate(dfs):

        count = (
            df
            .groupby("products")
            .size()
            .sort_values(ascending=False)
            .rename("Count clean rxns")
        )

        count_format = (
            count
            .value_counts()
            .reset_index()
            .rename(columns={"index": "number_of_paths",
                             "Count clean rxns":"frequency"})
        )

        ax[i].scatter(count_format.number_of_paths,
                      count_format.frequency,
                      marker=".",
                      color='k')

        ax[i].set_title(f"{names[i]}")
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].set_xlabel("Number of paths")
        ax[i].set_ylabel("Frequency")

    logger.log_fig(fig, "rxns_per_product")
    return fig


def top_k_products(
        df,
        k,
        name,
        figsize,
        logger
):
    '''Visualize top k products from a dataset using rdkit.

    Args:
        df: pd.DataFrame, Dataframe containing a column with products SMILES
        k: int, k top products to display
        name: str, name of the analyzed dataset
        figsize: tuple, size of the created figure

    '''

    import math

    #Get top k product smiles
    top_k_smiles = df.products.value_counts().index[:k].values
    top_k_counts = df.products.value_counts().values[:k]

    #To rdkit mol
    mols = [Chem.MolFromSmiles(i) for i in top_k_smiles]

    fig, axs = plt.subplots(math.ceil(k/2), 2, figsize=figsize)

    for i, mol in enumerate(mols):
        axs[i//2, i%2].imshow(Draw.MolToImage(mol))
        axs[i//2, i%2].set_axis_off()
        axs[i//2, i%2].set_title(top_k_counts[i], fontsize=6)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.suptitle(name + 'top_' + str(k) + '_products')
    plt.tight_layout()

    logger.log_fig(fig, f'{name}_top_{k}_products')


def properties_histogram(
        df_list,
        properties,
        names,
        logger,
        cat=True
):
    '''Plot histograms for product properties'''

    if cat:
        plot_func = single_property_cat
    else:
        plot_func = single_property_num

    for i, name  in enumerate(properties):
        # Make individual plot for each property
        dfs_counts = [df[name]
                      .value_counts()
                      .rename(names[i])
                      for i, df in enumerate(df_list)]

        plot_func(
            dfs_counts,
            name,
            names,
            True,
            logger
        )


def single_property_cat(
        series,
        prop_name,
        names,
        norm=False,
        logger=False,
        index_to_category=None
):
    """
    General function for plotting comparative barplot of
    a single CATEGORICAL property for a list of datasets.

    Args:
        s1, s2: Series containing .value_counts of property
        prop_name: name of the property, will be title of plot
        names: names of the dataset
        norm: normalize the barplot
        logger: use logger to log results
        index_to_category: dictionary mapping each category to a corresponding label
    """

    # Normalize series before joining
    if norm:
        for s in series:
            s /= s.sum()

    # Merge the datasets
    from functools import reduce
    merged = (
        reduce(lambda left, right:
               pd.merge(left, right,
                        left_index=True,
                        right_index=True,
                        how='outer'),
               series)
        .fillna(0)
        .sort_index()
        .reset_index()
    )

    # Plot only top-k, and a joint bar for the rest
    # top-k defined by both dfs.
    k = 30
    if merged.shape[0] > k:
        merged["sum"] = (
            merged
            .sum(axis=1)
            .sort_values(ascending=False)
        )

        # Sum of the tails
        rest_sum = merged.iloc[k:].sum()
        rest_sum["index"] = "others"
        merged = (
            pd.concat(
                [
                    merged.iloc[:k],
                    pd.DataFrame(rest_sum).T
                ],
            )
            .drop(columns="sum")
        )


    # Melt into single column for plotting
    melt = (
        pd.melt(
            merged,
            id_vars="index",
            value_name="freq",
            var_name="dataset"
        )
    )

    # Plot
    fig, ax = plt.subplots(figsize=(30,6))
    
    if index_to_category:
        melt['index'] = melt['index'].map(index_to_category)
    

    sns.barplot(
        data=melt,
        x="index",
        y="freq",
        hue="dataset",
        ax=ax
    )

    ax.set_xlabel(prop_name)

    logger.log_fig(fig, prop_name)


def single_property_num(
        series,
        prop_name,
        names,
        norm=False,
        logger=False
):
    """
    General function for plotting comparative barplot of
    a single NUMERICAL property for a list of datasets.

    Args:
        s1, s2: Numerical arrays.
        prop_name: name of the property, will be title of plot
        names: names of the dataset
    """

    fig, ax = plt.subplots()

    ax.hist(series, bins=50, label=names)
    ax.set_title(prop_name)
    ax.legend()
    ax.set_xlim(xmax=max(series[0]))

    logger.log_fig(fig, prop_name)




# TODO

def calculate_fingerprint(df, rxn):
    '''Calculate rxnfp for a given dataset
    '''
    #Take rxn smiles

    #Compute fp

    #compute minhash fps
    pass


def plot_TMAP(df, fps):
    '''Plot TMAP for a dataset

    Arg:
        -df: pd.DataFrame, calculated properties of the dataset to plot
        -fps: np.array, MinHash fingerprints computed for the reactions
    '''
    #compute rxnfp

    #Organize labels and properties
    pass


