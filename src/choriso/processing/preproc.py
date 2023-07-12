import os
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd
import requests
from rxn.chemutils.reaction_equation import (canonicalize_compounds,
                                             merge_reactants_and_agents,
                                             sort_compounds)
from rxn.chemutils.reaction_smiles import (ReactionFormat,
                                           parse_extended_reaction_smiles,
                                           to_reaction_smiles)
from tqdm.auto import tqdm

from choriso.processing import rxn_utils
from choriso.processing.logging import print

tqdm.pandas()

try:
    import jpype
    import jpype.imports

    try:
        jpype.startJVM("-Djava.awt.headlines=true",
                       classpath=['/opt/leadmine.jar'])
    except:
        pass
    from com.nextmovesoftware import leadmine

    extract = leadmine.LeadMine()
    leadmine_flag = True

except:
    leadmine_flag = False
    print("We don't have leadmine")


#Get correction dictionary
try:
    df = pd.read_csv('data/helper/corrected_leadmine.csv',
                     sep='\t',
                     header=None,
                     index_col=0)
    correct_dict = {row[0]: row[1].values[0] for row in df.iterrows()}
except:
    correct_dict = {}
    print('Correction dictionary not available')


def download_raw_data(data_dir="data/raw/"):
    """Download the raw CJHIF dataset."""

    url = 'https://drive.switch.ch/index.php/s/uthL9jTERVQJJoW/download'
    target_path = data_dir + 'raw_cjhif.tar.gz'

    if not os.path.isfile(target_path): # Only download if file doesn't already exist

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())

    # Decompress tar file
    with tarfile.open(data_dir + 'raw_cjhif.tar.gz') as f:
        f.extractall(data_dir)

    # Substitute "§" with tabs in extracted file
    subprocess.call([f"sed -i -e 's/§/\t/g' {data_dir}data_from_CJHIF_utf8"],
                    shell=True)

    # Load data
    raw_df = pd.read_csv(data_dir + 'data_from_CJHIF_utf8', sep='\t')

    return 0


def download_processed_data(data_dir="data/processed/"):
    '''Download processed data (after cleaning and atom mapping).
    '''

    base_url = 'https://drive.switch.ch/index.php/s/VaSVBCiXrmzYzGD/download?path=%2F&files={}'

    print('Downloading processed datasets...')

    files = ['cjhif.tar.gz', 'choriso.tar.gz', 'uspto.tar.gz']
    for fname in files:
        url = base_url.format(fname)

        target_path = data_dir + fname

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.isfile(target_path):

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.raw.read())

        with tarfile.open(data_dir + fname) as f:
            f.extractall(data_dir)


def get_structures_from_name(names, format_bond=True):
    '''Convert text with chemical structures to their corresponding SMILES
    using LeadMine.

    Args:
        names: str or set, text containing chemical structures
        format_bond: bool, use '~' to represent chemical species from the same compound
                     e.g: [Na+]~[Cl-] instead of [Na+].[Cl-]

    Returns:
        structures: set, chemical entities SMILES from text'''

    if leadmine_flag:

        names_list = names.split('|')

        def parse_entities(entity):
            '''Return SMILES from detected text entity'''
            # If the name is empty, return the same to mark the column
            if entity == 'empty':
                return 'empty'

            # If entity in `correct_dict`, correct
            elif entity in correct_dict.keys():
                return correct_dict[entity]

            # Else apply default LeadMine
            found = extract.findEntities(entity)
            smiles = [extract.resolveEntity(obj) for obj in found]

            if smiles: # If not empty
                return smiles[0]

        structures = {str(parse_entities(name)) for name in names_list}
        structures.discard("None")
        structures.discard("")

        if format_bond:
            structures = [structure.replace('.', '~') for structure in structures]

        structures = ".".join(structures)

        return structures

    else:
        print('Leadmine is not available')
        return 0

def column_check(row, original, new):
    '''Check if original and new columns contain the same number
    of elements.

    Args:
       row: pd.DataFrame row, row to check
       original: str, name of the original column
       new: str, name of the column that was created from the original

    Returns:
       match: bool, True if the number of element in both columns is equal
    '''

    #if the original column is empty, this is True
    if row[original] == '':
        match = True

    #If the new column is empty, this is False (leadmine didn't work)
    elif row[new] == '':
        match = False

    else:
        original = len(row[original].split('|'))
        new = len(row[new].split('.'))
        match = original == new

    return match


def preprocess_additives(
        data_dir,
        file_name,
        name="cjhif",
        logger=False
):
    '''First dataset preprocessing:
    Adapt additives information (solvents, catalysts, reagents).
        - Drop duplicates
        - Drop AMM and FG columns
        - Rename columns
        - Map additives' names to structures (SMILES) in new columns.
    '''

    #Create df with raw data
    cjhif = (pd.read_csv(data_dir + file_name,
                         sep='\t',
                         header = None,
                         )
             #Drop duplicate rows
             .drop_duplicates()
             #Fill NaN with empty strings
             .fillna('empty')
             #Drop columns 1 and 2
             .drop(columns=[1,2], axis=1)
             #Rename columns
             .rename(columns={0:'rxn_smiles',
                              3:'reagent',
                              4:'solvent',
                              5:'catalyst',
                              6:'yield'}
                     )
             )

    #Map reagent text to SMILES
    print('Getting reagent SMILES')
    cjhif['reagent_SMILES'] = cjhif['reagent'].progress_apply(get_structures_from_name)

    #Map solvent text to SMILES
    print('Getting solvent SMILES')
    cjhif['solvent_SMILES'] = cjhif['solvent'].progress_apply(get_structures_from_name)

    #Map catalyst text to SMILES
    print('Getting catalyst SMILES')
    cjhif['catalyst_SMILES'] = cjhif['catalyst'].progress_apply(get_structures_from_name)

    #Check if reagents and catalyst name have been correctly processed by Leadmine
    print('Checking reagent number')
    reagent_flag = cjhif.progress_apply(lambda x: column_check(x, 'reagent', 'reagent_SMILES'), axis=1)
    print('Checking catalyst number')
    catalyst_flag = cjhif.progress_apply(lambda x: column_check(x, 'catalyst', 'catalyst_SMILES'), axis=1)

    # Remove rows where text2smiles translation is faulty
    # Don't consider solvent in this
    filt = reagent_flag & catalyst_flag

    cjhif = cjhif[filt]

    if logger:
        logger.log({
            f"faulty rows text2smiles: {name}":(~filt).mean(),
            f"rows after additives preprocessing: {name}":len(cjhif)
        })

    return cjhif


def get_full_reaction_smiles(
        df,
        name="cjhif",
        logger=False
):
    '''Get full reaction SMILES from the preprocessed dataset by joining
    reagents and catalysts to the original reaction SMILES'''

    print('Generating full reaction smiles (including additives)')

    df['full_reaction_smiles'] = df.progress_apply(
        rxn_utils.join_additives,
        axis=1
    )

    if logger:
        logger.log({
            f"rows after join additives: {name}":len(df)
        })
    return df


def canonicalize_filter_reaction(
        df,
        column,
        name="cjhif",
        by_yield=False,
        logger=False
):
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

    #Canonicalize reaction SMILES, create new column for that
    df['canonic_rxn'] = (
        df[column]
        .progress_apply(lambda x:
                        rxn_utils.canonical_rxn(x)
                        )
    )

    #Drop invalid SMILES
    df = (
        df[df['canonic_rxn'] != 'Invalid SMILES']
        .reset_index(drop=True)
    )

    if by_yield:
        #take repeated reaction with highest yield
        high_duplicates = (
            df.iloc[
                df[df.duplicated(subset=['canonic_rxn'],
                                 keep=False)]
                .groupby('canonic_rxn')['yield']
                .idxmax()
                .values
            ]
        )

        #create clean df (no duplicated SMILES)
        filtered_df = pd.concat(
            [
                df.drop_duplicates('canonic_rxn',keep=False),
                high_duplicates
            ],
            ignore_index=True)

    else:
        filtered_df = df.drop_duplicates('canonic_rxn')


    if logger:
        logger.log({
            f"rows after canonicalization: {name}":len(df),
            f"rows after filter by yield: {name}":len(filtered_df)
        })

    return filtered_df


def clean_USPTO(
        df,
        logger=False
):
    '''Create canonical SMILES column for USPTO and clean it using the same function
    that was applied to out dataset
    '''
    print('Cleaning USPTO')

    def _canonical_rxn(rxn_smi):

        try:
            #set reaction type to use fragment bonds (~)
            rxn_type = ReactionFormat.STANDARD_WITH_TILDE

            #parse full reaction SMILES
            ReactEq = parse_extended_reaction_smiles(rxn_smi, remove_atom_maps=True)

            # If no reactants or no products, return invalid smiles
            if len(ReactEq.reactants)*len(ReactEq.products) == 0:
                return "Invalid SMILES"

            #Standard reaction: canonicalize reaction and sort compounds
            std_rxn = sort_compounds(canonicalize_compounds(merge_reactants_and_agents(ReactEq)))

            #Create final reaction SMILES
            rxn = to_reaction_smiles(std_rxn, rxn_type)

            return rxn
        except:
            return "Invalid SMILES"

    def _main_reactant(smiles):
        '''Auxiliary function to find main reactant from USPTO set'''

        reactants = parse_extended_reaction_smiles(smiles).reactants

        #remove non-atom characters
        simple_reacts = [re.sub(r'[\[\]\+\-.=()#~*@]', '' ,i) for i in reactants]

        #Take index of longest SMILES as a rough estimation of main reactant
        max_idx = np.argmax(np.array([len(i) for i in simple_reacts]))

        return reactants[max_idx]


    print("Canonicalizing USPTO")
    df['canonic_rxn'] = df['full_reaction_smiles'].progress_apply(lambda x: _canonical_rxn(x))
    df = (
        df[df['canonic_rxn'] != 'Invalid SMILES']
        .reset_index(drop=True)
    )
    print("Estimating main reactants in USPTO")
    df['main_reactant'] = df['full_reaction_smiles'].progress_apply(lambda x: _main_reactant(x))

    filtered_df = df.drop_duplicates('canonic_rxn')

    if logger:
        logger.log({
            f"rows after canonicalization: uspto":len(df),
            f"rows after filter by yield: uspto":len(filtered_df)
        })
    return filtered_df
