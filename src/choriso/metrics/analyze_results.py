'''General script to analyze benchmarking results implementing different metrics.'''

import os
import sys
import click
import pandas as pd
from choriso.metrics.selectivity import stereo_score, Evaluator
from tqdm import tqdm

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--chemistry', type=bool, default=True, help='Whether to compute chemistry metrics or not.')
@click.option('--mapping', type=bool, default=False, 
help='Whether to compute mapping and templates or not (these are required for chemistry metrics).')


def main(path, chemistry, mapping):

    files = sorted(os.listdir(path))

    #check if results.txt already exists
    if 'results.txt' in files:
        files.remove('results.txt')
    if 'results.csv' in files:
        files.remove('results.csv')

    #write results to file
    with open(os.path.join(path, 'results.txt'), 'w') as f:

        #create df to store results
        df = pd.DataFrame(columns=['top-1', 'top-2', 'top-5', 'regio', 'stereo'])

        #write LATeX table header
        f.write(r'\begin{tabular}{|| c | c | c | c | c | c | c ||}')
        f.write('\n')
        f.write(r'\hline')
        f.write('\n')
        f.write(r'model & top-1 & top-2 & top-5 & regio & stereo \\')
        f.write('\n')
        f.write(r'\hline\hline')
        f.write('\n')
        
        for file in tqdm(files):

            #use evaluator to compute metrics
            evaluator = Evaluator(os.path.join(path, file), mapping=mapping)
            evaluator.compute_metrics(chemistry=chemistry)

            top_1 = evaluator.metrics['top-1']
            top_2 = evaluator.metrics['top-2']
            top_5 = evaluator.metrics['top-5']
            
            if chemistry:
                regio = evaluator.metrics['regio_score'][0]
                stereo = evaluator.metrics['stereo_score'][0]

                
            #write results to Latex table
            name = file[:-4].replace('_', ' ')
            
            if chemistry:
                f.write(f'{name} & {top_1} & {top_2} & {top_5} & {regio} & {stereo} \\\\  [1ex]')
                f.write('\n')
                f.write(r'\hline')
                f.write('\n')
                
                df.loc[file] = [top_1, top_2, top_5, regio, stereo]
            else:
                f.write(f'{name} & {top_1} & {top_2} & {top_5} \\\\  [1ex]')
                f.write('\n')
                f.write(r'\hline')
                f.write('\n')

                df.loc[file] = [top_1, top_2, top_5, '', '']
  
            
        f.write(r'\end{tabular}')

        df.to_csv(os.path.join(path, 'results.csv'), index=False)


if __name__ == '__main__':
    main()

