"""General script to analyze benchmarking results implementing different metrics."""

import os
import sys

import click
import pandas as pd
from tqdm import tqdm

from choriso.metrics.selectivity import (
    co2_transform,
    flag_regio_problem,
    flag_stereo_problem,
    regio_score,
    stereo_score,
    top_n_accuracy,
)


def extract_results(names):
    """Extract the results from the folders and save them in a csv file in the 'predictions' folder
    for subsequent analysis. The results are saved in a csv file with the name of the model.

    Args:
        names (list): list of folders containing the results of the models

    """

    if not os.path.exists("results/predictions"):
        os.makedirs("results/predictions")
    if not os.path.exists("results/sustainability"):
        os.makedirs("results/sustainability")

    for name in names:
        # First, walk through each directory and locate the subfolders with the results
        folders = list(os.walk(name))[0][1]  # list of subfolders

        # Then, for each folder, extract the results if they contain the subfolder 'results'
        for folder in folders:
            path = os.path.join(name, folder)

            if "results" in os.listdir(path):
                # Extract the results if the path exists
                if os.path.exists(
                    os.path.join(path, "results/all_results.csv")
                ):
                    df = pd.read_csv(
                        os.path.join(path, "results/all_results.csv")
                    )

                    # select only 'canonical_rxn', 'target', 'pred_0', 'pred_1' columns and templates (if available)
                    defaults = [
                        "canonical_rxn",
                        "target",
                        "pred_0",
                        "pred_1",
                        "pred_2",
                    ]
                    templates_mapped = [
                        "rxnmapper_aam",
                        "template_r0",
                        "template_r1",
                        "split",
                    ]

                    if not all(elem in df.columns for elem in defaults):
                        print(
                            "all_results.csv does not contain the required columns"
                        )
                        continue

                    if all(elem in df.columns for elem in templates_mapped):
                        defaults.extend(templates_mapped)
                        df = df[defaults]

                    else:
                        # try to extract the mapping and the templates from the original test.tsv file
                        if os.path.exists(
                            os.path.join(f"data/{folder}", "test.tsv")
                        ):
                            print(
                                f"Retrieving templates and mapping from data/{folder}/test.tsv"
                            )
                            df_test = pd.read_csv(
                                os.path.join(f"data/{folder}", "test.tsv"),
                                sep="\t",
                            )
                            if all(
                                elem in df_test.columns
                                for elem in templates_mapped
                            ):
                                # add the columns to the dataframe df
                                df = pd.concat(
                                    (df[defaults], df_test[templates_mapped]),
                                    axis=1,
                                )

                            else:
                                print(
                                    "test.tsv does not contain the required columns"
                                )
                                continue

                        else:
                            print("Mapping and templates not found")
                            df = df[defaults]

                    # Save the results in 'predictions' folder renaming the file with the name of the model
                    df.to_csv(
                        "results/predictions/" + name + "_" + folder + ".csv",
                        index=False,
                    )

                    emission_file = os.path.join(
                        path, "results/predict_emission.csv"
                    )
                    train_file = os.path.join(
                        path, "results/train_emission.csv"
                    )

                    if os.path.exists(emission_file) and os.path.exists(
                        train_file
                    ):
                        # Read the CO2 CSV files into dataframes
                        predict_df = pd.read_csv(
                            name
                            + "/"
                            + folder
                            + "/results/predict_emission.csv",
                            index_col=0,
                        )
                        preprocess_df = pd.read_csv(
                            name
                            + "/"
                            + folder
                            + "/results/preprocess_emission.csv",
                            index_col=0,
                        )
                        train_df = pd.read_csv(
                            name
                            + "/"
                            + folder
                            + "/results/train_emission.csv",
                            index_col=0,
                        )

                        # Add a new column to each dataframe to store the original filename
                        predict_df["Source"] = "predict"
                        preprocess_df["Source"] = "preprocess"
                        train_df["Source"] = "train"

                        # Concatenate the dataframes (only last result in case we have multiple runs)
                        merged_df = pd.concat(
                            [
                                predict_df.tail(1),
                                preprocess_df.tail(1),
                                train_df.tail(1),
                            ]
                        )

                        # save the merged dataframe to a csv file in results/co2 folder
                        merged_df.to_csv(
                            "results/sustainability/"
                            + name
                            + "_"
                            + folder
                            + ".csv",
                            index=False,
                        )
                    else:
                        print("No data on CO2 emissions found in " + path)

                else:
                    print(f"No results found in {path}")


def compute_flags(path):
    """Compute the flags for the predictions

    Args:
        path (str): path to the folder containing the results of the models

    """

    results_path = os.path.join(path, "predictions")

    files = sorted(os.listdir(results_path))

    # don't consider results.txt and results.csv
    files = [
        file for file in files if file not in ["results.txt", "results.csv"]
    ]

    for file in tqdm(files):
        # read csv file
        df = pd.read_csv(os.path.join(results_path, file))

        # replace nan with False
        df = df.fillna(False)

        # compute flags for regio and stereo if not present
        if "regio_flag" not in df.columns:
            df["regio_flag"] = df.apply(
                lambda x: flag_regio_problem(
                    x["canonical_rxn"], (x["rxnmapper_aam"], x["template_r1"])
                ),
                axis=1,
            )

        if "stereo_flag" not in df.columns:
            df["stereo_flag"] = df["template_r0"].parallel_apply(
                lambda x: flag_stereo_problem(x)
            )

        df.to_csv(os.path.join(results_path, file), index=False)


def compute_results(path):
    """Compute the results of the models in the path and save them in a txt file.

    Args:
        path (str): path to the folder containing the results of the models
        chemistry (bool): whether the models are chemistry models or not
        mapping (str): mapping to use to compute the metrics
    """

    # First compute metrics without CO2
    results_path = os.path.join(path, "predictions")

    files = sorted(os.listdir(results_path))

    # check if results.txt already exists
    if "results.txt" in files:
        files.remove("results.txt")
    if "results.csv" in files:
        files.remove("results.csv")

    # write results to file
    with open(os.path.join(results_path, "results.txt"), "w") as f:
        # create df to store results
        big_df = pd.DataFrame(columns=["top-1", "top-2", "stereo", "regio"])

        # write LATeX table header
        f.write(r"\begin{tabular}{|| c | c | c | c | c | c ||}")
        f.write("\n")
        f.write(r"\hline")
        f.write("\n")
        f.write(r"model & top-1 & top-2 & stereo & regio \\")
        f.write("\n")
        f.write(r"\hline\hline")
        f.write("\n")

        for file in tqdm(files):
            # read csv file
            df = pd.read_csv(os.path.join(results_path, file))

            for split in df["split"].unique():
                df_sp = df[df["split"] == split]
                top_1 = top_n_accuracy(df_sp, 1)
                top_2 = top_n_accuracy(df_sp, 2)
                regio = regio_score(df_sp)
                stereo = stereo_score(df_sp)

                # write results to Latex table
                name = file[:-4].replace("_", " ")

                f.write(
                    f"{name + '_' + split} & {top_1} & {top_2} & {stereo} & {regio} \\\\  [1ex]"
                )
                f.write("\n")
                f.write(r"\hline")
                f.write("\n")

                # write results to df where the index of the row is the model name
                big_df.loc[f"{file[:-4]}_{split}"] = [
                    top_1,
                    top_2,
                    stereo,
                    regio,
                ]

        f.write(r"\end{tabular}")

        big_df.to_csv(os.path.join(results_path, "results.csv"))

    # now compute co2
    sustainability_path = os.path.join(path, "sustainability")

    sust_files = sorted(os.listdir(sustainability_path))

    if "sustainability_prediction.csv" in sust_files:
        sust_files.remove("sustainability_prediction.csv")
    if "sustainability_train.csv" in sust_files:
        sust_files.remove("sustainability_train.csv")

    # create a df to store predictions and another to store train
    sust_columns = [
        "duration(s)",
        "duration scaled",
        "CO2_emissions(kg)",
        "co2_scaled",
    ]

    df_pred = pd.DataFrame(columns=sust_columns)
    df_train = pd.DataFrame(columns=sust_columns)

    for file in sust_files:
        df = pd.read_csv(os.path.join(sustainability_path, file))

        # extract data for pred
        co2 = df.loc[0, "CO2_emissions(kg)"]
        co2_scaled = co2_transform(co2)
        time = df.loc[0, "duration(s)"] / 3600
        time_scaled = co2_transform(time, mode="time")

        # add data to df_pred as a new row
        df_pred.loc[file[:-4]] = [time, time_scaled, co2, co2_scaled]

        # extract data for train
        co2 = df.loc[2, "CO2_emissions(kg)"]
        co2_scaled = co2_transform(co2)
        time = df.loc[2, "duration(s)"] / 3600
        time_scaled = co2_transform(time, mode="time")

        # add data to df_train as a new row
        df_train.loc[file[:-4]] = [time, time_scaled, co2, co2_scaled]

    # round all values to 2 decimals and save
    df_pred = df_pred.round(2)
    df_pred.to_csv(
        os.path.join(sustainability_path, "sustainability_prediction.csv")
    )

    df_train = df_train.round(2)
    df_train.to_csv(
        os.path.join(sustainability_path, "sustainability_train.csv")
    )


@click.command()
@click.option("--results_folders", "-r", type=str, multiple=True)
@click.option("--path", type=str, default="results")
def main(results_folders, path):
    """Main results analysis pipeline for the metrics.
    Args:
        results_folders: str, path to results, if previous results exist, load them from this path
        path: str, compute results and store them in this path
        chemistry: bool. whether the model is a chemistry model or not
        mapping: bool. whether to compute mapping or not
    """

    if results_folders:
        if not os.path.exists("results/predictions"):
            print(
                f"Extracting results from {len(results_folders)} folder(s)..."
            )
            extract_results(results_folders)

    if path:
        print("Computing flags...")
        compute_flags(path)
        print("Computing results...")
        compute_results(path)


if __name__ == "__main__":
    main()
