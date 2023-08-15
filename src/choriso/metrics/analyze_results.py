'''General script to analyze benchmarking results implementing different metrics.'''
import os
import sys
import click
import pandas as pd
from choriso.metrics.metrics.selectivity import Evaluator
from tqdm import tqdm


def extract_results(names):
    '''Extract the results from the folders and save them in a csv file in the 'predictions' folder
    for subsequent analysis. The results are saved in a csv file with the name of the model.
    
    Args:
        names (list): list of folders containing the results of the models
    
    '''

    if not os.path.exists('results/predictions'):
        os.mkdir('results/predictions')
    if not os.path.exists('results/co2'):
        os.mkdir('results/co2')

    for name in names:
        #First, walk through each directory and locate the subfolders with the results
        folders = list(os.walk(name))[0][1]

        #Then, for each folder, extract the results if they contain the subfolder 'results'
        for folder in folders:
            path = os.path.join(name, folder)

            if 'results' in os.listdir(path):
                #Extract the results if the path exists
                if os.path.exists(os.path.join(path, 'results/all_results.csv')):

                    df = pd.read_csv(os.path.join(path, 'results/all_results.csv'))

                    #select only 'canonical_rxn', 'target', 'pred_0', 'pred_1' columns and templates (if available)
                    if 'template_r0' in df.columns:
                        df = df[['canonical_rxn', 'target', 'pred_0', 'pred_1', 'mapped_rxn', 'template_r0', 'template_r1']]
                    else:
                        df = df[['canonical_rxn', 'target', 'pred_0', 'pred_1']]
                    
                    #Save the results in 'predictions' folder renaming the file with the name of the model
                    df.to_csv('results/predictions/' + name + '_' + folder + '.csv', index=False)

                    # Read the CO2 CSV files into dataframes
                    predict_df = pd.read_csv(name + '/' + folder + '/results/predict_emission.csv', index_col=0)
                    preprocess_df = pd.read_csv(name + '/' + folder + '/results/preprocess_emission.csv', index_col=0)
                    train_df = pd.read_csv(name + '/' + folder + '/results/train_emission.csv', index_col=0)

                    # Add a new column to each dataframe to store the original filename
                    predict_df['Source'] = 'predict'
                    preprocess_df['Source'] = 'preprocess'
                    train_df['Source'] = 'train'

                    # Concatenate the dataframes
                    merged_df = pd.concat([predict_df, preprocess_df, train_df])

                    #save the merged dataframe to a csv file in results/co2 folder
                    merged_df.to_csv('results/co2/' + name + '_' + folder + '.csv', index=False)

                else:
                    print('No results in ' + path)


#use a list of paths and extract file
def compute_results(path, chemistry, mapping):
    '''Compute the results of the models in the path and save them in a txt file.

    Args:
        path (str): path to the folder containing the results of the models
        chemistry (bool): whether the models are chemistry models or not
        mapping (str): mapping to use to compute the metrics
    '''

    #First compute metrics without CO2

    results_path = os.path.join(path, 'predictions')

    files = sorted(os.listdir(results_path))

    #check if results.txt already exists
    if 'results.txt' in files:
        files.remove('results.txt')
    if 'results.csv' in files:
        files.remove('results.csv')
    
    #write results to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as f:

        #create df to store results
        df = pd.DataFrame(columns=['top-1', 'top-2', 'stereo', 'regio'])

        #write LATeX table header
        f.write(r'\begin{tabular}{|| c | c | c | c | c | c ||}')
        f.write('\n')
        f.write(r'\hline')
        f.write('\n')
        f.write(r'model & top-1 & top-2 & stereo & regio \\')
        f.write('\n')
        f.write(r'\hline\hline')
        f.write('\n')
        
        for file in tqdm(files):

            #use evaluator to compute metrics
            evaluator = Evaluator(os.path.join(results_path, file), mapping=mapping, sample=False, save=True)
            evaluator.compute_metrics(chemistry=chemistry)

            top_1 = evaluator.metrics['top-1']
            top_2 = evaluator.metrics['top-2']
            
            if chemistry:
                regio = evaluator.metrics['regio_score'][0]
                stereo = evaluator.metrics['stereo_score'][0]

                
            #write results to Latex table
            name = file[:-4].replace('_', ' ')
            
            if chemistry:
                f.write(f'{name} & {top_1} & {top_2} & {stereo} & {regio} \\\\  [1ex]')
                f.write('\n')
                f.write(r'\hline')
                f.write('\n')
                
                #write results to df where the index of the row is the model name
                df.loc[file[:-4]] = [top_1, top_2, stereo, regio]

            else:
                f.write(f'{name} & {top_1} & {top_2} \\\\  [1ex]')
                f.write('\n')
                f.write(r'\hline')
                f.write('\n')

                df.loc[file[:-4]] = [top_1, top_2, '', '']
  
            
        f.write(r'\end{tabular}')

        df.to_csv(os.path.join(results_path, 'results.csv'))

    # co2 = []

    # #move files that end with _CO2.csv to co2
    # for file in files:
    #     if file.endswith('_CO2.csv'):
    #         co2.append(file)
    #         files.remove(file)

    # if 'co2.csv' in files:
    #     files.remove('co2.csv')
    # if 'co2.txt' in files:
    #     files.remove('co2.txt')

    # with open(os.path.join(path, 'co2.txt'), 'w') as f:

    #     #merge all the .csv files in the co2 list into one with the same columns
    #     co2_df = pd.DataFrame(columns=['duration(s)','power_consumption(kWh)','CO2_emissions(kg)','co2_scaled','kwh_scaled'])
    #     #write LATeX table header
    #     f.write(r'\begin{tabular}{|| c | c | c | c | c ||}')
    #     f.write('\n')
    #     f.write(r'\hline')
    #     f.write('\n')
    #     f.write(r'model & CO2 (kg) & CO2 scaled & Energy (kWh) & Energy scaled \\')
    #     f.write('\n')
    #     f.write(r'\hline\hline')
    #     f.write('\n') 

    #     for file in co2:
    #         df = pd.read_csv(os.path.join(path, file))
    #         #set index of row 0 to the file name
    #         df.index = [file[:-4]]
    #         co2_df = pd.concat([co2_df, df], axis=0)
    #         power = df['power_consumption(kWh)'][0].round(2)
    #         co2 = df['CO2_emissions(kg)'][0].round(2)
    #         co2_scaled = df['co2_scaled'][0].round(2)
    #         kwh_scaled = df['kwh_scaled'][0].round(2)

    #         #write results to Latex table
    #         name = file[:-4].replace('_', ' ')
    #         f.write(f'{name} & {co2} & {co2_scaled} & {power} & {kwh_scaled} \\\\  [1ex]')
    #         f.write('\n')
    #         f.write(r'\hline')
    #         f.write('\n')

    #     #round all values to 2 decimals
    #     co2_df = co2_df.round(2)
    #     co2_df.to_csv(os.path.join(path, 'co2.csv'))

@click.command()
@click.option('--results_folders', '-r', type=str, multiple=True)
@click.option('--path', type=click.Path(exists=True), default='results')
@click.option('--chemistry', type=bool, default=True, help='Whether to compute chemistry metrics or not.')
@click.option('--mapping', type=bool, default=False, 
help='Whether to compute mapping and templates or not (these are required for chemistry metrics).')

def main(results_folders, path, chemistry, mapping):
    if results_folders:
        print('Extracting results from folders...')
        extract_results(results_folders)
    
    if path:
        compute_results(path, chemistry, mapping)

if __name__ == '__main__':
    main()

