"""This module contains the code for selectivity metrics"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction
from transformers import logging
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=24)

logging.set_verbosity_error()  # Only log errors


def aam_from_smiles(list_rxn_smiles):
    '''Get attention guided atom maps from a list of reaction SMILES.
    Args:
        list_rxn_smiles: list, reaction SMILES
    Out:
        out: list, attention guided atom maps
    '''
    rxn_mapper = RXNMapper()
    out = []
    for i, rxn in enumerate(list_rxn_smiles):
        try:
            out.append(rxn_mapper.get_attention_guided_atom_maps([rxn])[0])
        except: 
            out.append({"confidence": "", "mapped_rxn": ""})
            print('Issue with reaction', i, rxn)
    return(out)


def template_smarts_from_mapped_smiles(
        mapped_smiles,
        radius=0
):
    '''Get reaction template from mapped reaction SMILES.
    Args:
        mapped_smiles: str, mapped reaction SMILES
        radius: int, radius of the reaction template
        
    Out:
        template: str, reaction template
    '''

    try:
        rxn = ChemicalReaction(mapped_smiles, clean_smiles = False)
        rxn.generate_reaction_template(radius)
        return(rxn.canonical_template.smarts)
    
    except:
        return("")


def flag_regio_problem(rxn):
    '''Flag regioselectivity problems. For the moment only one-product
    reactions. The function extracts the reaction template (only reacting atoms) and then checks
    if the matching atoms in the product can generate several products.
    
    Args:
        rxn: str, reaction SMILES
    
    Out:
        bool, True if the reaction is regioselective, False otherwise
    '''

    def _sanitize_filter_prods(prods):
        good = []
        for prod in prods:
            try: 
                x = Chem.SanitizeMol(prod[0])
                good.append(Chem.MolToSmiles(prod[0]))
            except:
                pass
        return set(good)

    #extract rxn template
    map_rxn = aam_from_smiles([rxn])[0]['mapped_rxn']
    template = template_smarts_from_mapped_smiles(map_rxn, radius=1)

    if template:

        products = rxn.split('>>')[1]

        if '@' in products:
            return False
        
        rxn_list = rxn.split('>>')[0].split('.')
        map_list = map_rxn.split('>>')[0].split('.')

        #compare lists and extract only the elements from rxn_list that are different from rxn_smiles_list
        reactants = [i for i in rxn_list if i not in map_list]

        #check if reactants generate several products
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


def flag_stereo_problem(rxn):
    '''Flag stereoselectivity problems.
    Args:
        rxn: str, reaction SMILES
    
    Out:
        bool, True if the reaction has stereoselectivity issues, False otherwise
    '''

    #extract rxn template
    map_rxn = aam_from_smiles([rxn])[0]['mapped_rxn']
    template = template_smarts_from_mapped_smiles(map_rxn)

    if template:

        try:
            temp_prods = template.split('>>')[1].split('.')
            #check if any of the strings in prods contain '@'
            if any('@' in prod for prod in temp_prods):
                return True
        except:
            return False

#Regio
def regio_score(y_true, y_pred):
    """Regioselectivity classification score. Regiochemistry is defined supposing that the
    outcome of a reaction can be any of two isomers. Regio = preference for bonding one 
    atom over another.
    
    Args:
        y_true: array, true reaction SMILES
        y_pred: array, predicted reaction SMILES
    
    Out:
        acc: int, accuracy for regiochemistry reactions
    """
    #flag reactions with regiochem problems
    flags = [flag_regio_problem(rxn) for rxn in y_true]
    print(flags)
    y_true = y_true[flags]
    y_pred = y_pred[flags]

    #check if products are the same
    true_prods = np.array([i.split('>>')[1] for i in y_true])
    pred_prods = np.array([i.split('>>')[1] for i in y_pred])

    acc = true_prods == pred_prods
    acc = np.sum(acc)/len(acc)

    return acc


#Stereo
#Consider only reactions with stereocenters
def stereo_score(df):
    '''Stereoselectivity classification score. Top1 accuracy for stereoselectivity.

    Args:
        df: dataframe, dataframe with columns 'canonical_rxn', 'target', 'pred_0'

    Out:
        acc: int, accuracy for stereoselectivity reactions

    '''

    #flag reactions with stereochem problems
    df['flags'] = df['canonical_rxn'].apply(flag_stereo_problem)
    df = df[df['flags'] == True]

    #check if products are the same
    true_prods = np.array(df['target'].values)

    #predicted products
    pred_prods = np.array(df['pred_0'].values)

    acc = true_prods == pred_prods
    acc = np.sum(acc)/len(acc)

    return acc




class Evaluator():

    '''Evaluator class for reaction prediction models. It contains functions to evaluate
    the performance of the model in terms of accuracy and chemistry-specific metrics.
    '''

    def __init__(self, file, mapping=False, sample=False, save=False):

        self.file_path = file
        self.file = pd.read_csv(file)
        if sample:
            self.file = self.file.sample(1000, random_state=33)
        self.mapping = mapping
        self.metrics = {}
        self.save = save
        
        #check if file has mapped_rxn column, and if not, create it and save it
        if self.mapping == True:
            if 'mapped_rxn' not in self.file.columns:
                print('Mapping reactions...')
                maps = aam_from_smiles(self.file['canonical_rxn'].values)
                self.file['mapped_rxn'] = [i['mapped_rxn'] for i in maps]
                if self.save == True:
                    self.file.to_csv(file, index=False)
            print('Extracting templates...')
            self.file['template_r0'] = self.file['mapped_rxn'].parallel_apply(lambda x: template_smarts_from_mapped_smiles(x))
            self.file['template_r1'] = self.file['mapped_rxn'].parallel_apply(lambda x: template_smarts_from_mapped_smiles(x, radius=1))
            if self.save == True:
                self.file.to_csv(file, index=False)
            self.mapping = False

    def top_n_accuracy(self, df, n):
        '''Take predicition dataframe and compute top-n accuracy
        
        Args:
            df (pd.DataFrame): dataframe with predictions
            n (int): number of top predictions to consider

        Returns:
            float: top-n accuracy
        '''

        correct = 0

        for i, row in df.iterrows():
            
            for i in range(n):
                if row['target'] == row[f'pred_{i}']:
                    correct += 1
                    break
        
        accuracy = round(correct / len(df)*100, 1)

        return accuracy


    def flag_regio_problem(self, rxn, map_rxn, template):
        '''Flag regioselectivity problems.
        Args:
            rxn: str, reaction SMILES
            map_rxn: dict, mapped reaction
            template: str, extracted template with radius=1 from reaction SMILES
            
        Out:
            bool, True if the reaction has regioselectivity issues, False otherwise
        '''

        def _sanitize_filter_prods(prods):
            good = []
            for prod in prods:
                try: 
                    x = Chem.SanitizeMol(prod[0])
                    good.append(Chem.MolToSmiles(prod[0]))
                except:
                    pass
            return set(good)
        
        #proceed only if template is not nan
        
        if pd.isna(template) == False:

            products = rxn.split('>>')[1]
            #False in case we are counting stereochemistry
            if '@' in products:
                return False
    
            rxn_list = rxn.split('>>')[0].split('.')
            map_list = map_rxn.split('>>')[0].split('.')

            #compare lists and extract only the elements from rxn_list that are different from rxn_smiles_list
            reactants = [i for i in rxn_list if i not in map_list]
            
            #check if reactants generate several products
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


    def regio_score(self, negative_acc=False):
        '''Regioselectivity classification score. Top1 accuracy for reactions with regioselectivity
        problems.
        
        Args:
            df: pd.DataFrame, dataframe with predictions
            
        Returns:
            acc: float, top-1 accuracy for reactions where regioselectivity is an issue
        '''

        if 'regio_flag' not in self.file.columns:
            #flag reactions with regiochem problems
            self.file['regio_flag'] = self.file.parallel_apply(lambda x: self.flag_regio_problem(x['canonical_rxn'], x['mapped_rxn'], x['template_r1']), axis=1)
            if self.save == True:
                self.file.to_csv(self.file_path, index=False)
        
        df_true = self.file[self.file['regio_flag'] == True]
        df_false = self.file[self.file['regio_flag'] == False]

        #check if products are the same
        true_prods = np.array(df_true['target'].values)
        pred_prods = np.array(df_true['pred_0'].values)

        acc = true_prods == pred_prods
        acc = round(np.sum(acc)/len(acc)*100, 1)

        if negative_acc == True:
            #check if products are the same
            true_prods = np.array(df_false['target'].values)
            pred_prods = np.array(df_false['pred_0'].values)

            acc_neg = true_prods == pred_prods
            acc_neg = round(np.sum(acc_neg)/len(acc_neg)*100, 1)

            return acc, acc_neg
        
        else:
            return acc
        

    def flag_stereo_problem(self, template):
        '''Flag stereoselectivity problems.
        Args:
            template: str, extracted template with radius=0 from reaction SMILES
        
        Out:
            bool, True if the reaction has stereoselectivity issues, False otherwise
        '''

        if template:

            try:
                temp_prods = template.split('>>')[1].split('.')
                #check if any of the strings in prods contain '@'
                if any('@' in prod for prod in temp_prods):
                    return True

                else:
                    return False
            except:
                return False


    def stereo_score(self, negative_acc=False):
        '''Stereoselectivity classification score. Top1 accuracy for reactions with stereoselectivity
        problems.

        Args:
            df: pd.DataFrame, dataframe with predictions

        Returns:
            acc: float, top-1 accuracy for reactions where stereoselectivity is an issue
            negative_acc: float, top-1 accuracy for reactions where stereoselectivity is not an issue
        '''

        if 'stereo_flag' not in self.file.columns:
            #flag reactions with stereochem problems
            self.file['stereo_flag'] = self.file['template_r0'].apply(self.flag_stereo_problem)
            if self.save == True:
                self.file.to_csv(self.file_path, index=False)

        df_true = self.file[self.file['stereo_flag'] == True]
        df_false = self.file[self.file['stereo_flag'] == False]

        #check if products are the same
        true_prods = np.array(df_true['target'].values) 
        pred_prods = np.array(df_true['pred_0'].values)

        acc = true_prods == pred_prods
        acc = round(np.sum(acc)/len(acc)*100, 1)

        if negative_acc:
            #check if products are the same
            true_prods = np.array(df_false['target'].values)
            pred_prods = np.array(df_false['pred_0'].values)

            acc_neg = true_prods == pred_prods
            acc_neg = round(np.sum(acc_neg)/len(acc_neg), 1)

            return acc, acc_neg

        else:
            return acc


    def compute_metrics(self, chemistry=True):
        '''Compute all metrics for the predictions file and store them in self.metrics dict
        
        Args:
            chemistry: bool, True if compute chemistry-specific metrics
        '''

        self.metrics['top-1'] = self.top_n_accuracy(self.file, 1)
        self.metrics['top-2'] = self.top_n_accuracy(self.file, 2)
        self.metrics['top-5'] = self.top_n_accuracy(self.file, 5)

        if chemistry:
            self.metrics['stereo_score'] = self.stereo_score(negative_acc=True)
            self.metrics['regio_score'] = self.regio_score(negative_acc=True)

        return self


if __name__ == '__main__':
    
    rxnmapper = RXNMapper()
    
    evaluator = Evaluator('data/predictions/transformer.csv', mapping=True, sample=True)

    print(evaluator.file.head())
    evaluator.compute_metrics()
    print(evaluator.metrics)

    print(evaluator.file.head())
    # rxn = 'CCO.C[C@]12CCC3(CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12)OCCO3.[Ni]>>C[C@]12CC[C@H]3[C@@H](CC=C4CC5(CC[C@@]43C)OCCO5)[C@@H]1CC[C@@H]2O'
    # rxn1 = 'CCO.CCOC(=O)CCCCCC[C@H]1CC(=O)N[C@@H]1C(=O)OC.O=C([O-])[O-].[K+].[K+]>>CCOC(=O)CCCCCC[C@@H]1CC(=O)N[C@H]1C(=O)O'
    # rxn2 = 'C1CCC(CC)C=C1.Br>>C1CCC(CC)C(Br)C1'
    # rxn3 = 'C1=CC2C=CC1C2.C=C(NC(C)=O)c1ccc(OC)cc1.CO.[H][H]>>COc1ccc([C@@H](C)NC(C)=O)cc1'
    
    # print(flag_regio_problem(rxn))
    # print(flag_regio_problem(rxn1))
    # print(flag_regio_problem(rxn2))
    # print(flag_regio_problem(rxn3))

