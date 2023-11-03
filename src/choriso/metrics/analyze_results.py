"""General script to analyze benchmarking results implementing different metrics."""

import os
import sys

import click
import pandas as pd
from tqdm import tqdm

from choriso.metrics.metrics.selectivity import Evaluator, co2_transform


def extract_results(names):
    """Extract the results from the folders and save them in a csv file in the 'predictions' folder
    for subsequent analysis. The results are saved in a csv file with the name of the model.

    Args:
        names (list): list of folders containing the results of the models

    """

    if not os.path.exists("results/predictions"):
        os.mkdir("results/predictions")
    if not os.path.exists("results/sustainability"):
        os.mkdir("results/sustainability")

    for name in names:
        # First, walk through each directory and locate the subfolders with the results
        folders = list(os.walk(name))[0][1]

        # Then, for each folder, extract the results if they contain the subfolder 'results'
        for folder in folders:
            path = os.path.join(name, folder)

            if "results" in os.listdir(path):
                # Extract the results if the path exists
                if os.path.exists(os.path.join(path, "results/all_results.csv")):
                    df = pd.read_csv(os.path.join(path, "results/all_results.csv"))

                    # select only 'canonical_rxn', 'target', 'pred_0', 'pred_1' columns and templates (if available)
                    if "template_r0" in df.columns:
                        df = df[
                            [
                                "canonical_rxn",
                                "target",
                                "pred_0",
                                "pred_1",
                                "pred_2",
                                "mapped_rxn",
                                "template_r0",
                                "template_r1",
                            ]
                        ]
                    else:
                        df = df[["canonical_rxn", "target", "pred_0", "pred_1", "pred_2"]]

                    # Save the results in 'predictions' folder renaming the file with the name of the model
                    df.to_csv("results/predictions/" + name + "_" + folder + ".csv", index=False)

                    # Read the CO2 CSV files into dataframes
                    predict_df = pd.read_csv(
                        name + "/" + folder + "/results/predict_emission.csv", index_col=0
                    )
                    preprocess_df = pd.read_csv(
                        name + "/" + folder + "/results/preprocess_emission.csv", index_col=0
                    )
                    train_df = pd.read_csv(
                        name + "/" + folder + "/results/train_emission.csv", index_col=0
                    )

                    # Add a new column to each dataframe to store the original filename
                    predict_df["Source"] = "predict"
                    preprocess_df["Source"] = "preprocess"
                    train_df["Source"] = "train"

                    # Concatenate the dataframes (only last result in case we have multiple runs)
                    merged_df = pd.concat(
                        [predict_df.tail(1), preprocess_df.tail(1), train_df.tail(1)]
                    )

                    # save the merged dataframe to a csv file in results/co2 folder
                    merged_df.to_csv(
                        "results/sustainability/" + name + "_" + folder + ".csv", index=False
                    )

                else:
                    print("No results in " + path)


# use a list of paths and extract file
def compute_results(path, chemistry, mapping):
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
        df = pd.DataFrame(columns=["top-1", "top-2", "stereo", "regio"])

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
            print(file)
            # use evaluator to compute metrics
            evaluator = Evaluator(
                os.path.join(results_path, file), mapping=mapping, sample=False, save=True
            )
            evaluator.compute_metrics(chemistry=chemistry)

            top_1 = evaluator.metrics["top-1"]
            top_2 = evaluator.metrics["top-2"]

            if chemistry:
                regio = evaluator.metrics["regio_score"][0]
                stereo = evaluator.metrics["stereo_score"][0]

            # write results to Latex table
            name = file[:-4].replace("_", " ")

            if chemistry:
                f.write(f"{name} & {top_1} & {top_2} & {stereo} & {regio} \\\\  [1ex]")
                f.write("\n")
                f.write(r"\hline")
                f.write("\n")

                # write results to df where the index of the row is the model name
                df.loc[file[:-4]] = [top_1, top_2, stereo, regio]

            else:
                f.write(f"{name} & {top_1} & {top_2} \\\\  [1ex]")
                f.write("\n")
                f.write(r"\hline")
                f.write("\n")

                df.loc[file[:-4]] = [top_1, top_2, "", ""]

        f.write(r"\end{tabular}")

        df.to_csv(os.path.join(results_path, "results.csv"))

    # now compute co2
    sustainability_path = os.path.join(path, "sustainability")

    sust_files = sorted(os.listdir(sustainability_path))

    if "sustainability_prediction.csv" in sust_files:
        sust_files.remove("sustainability_prediction.csv")
    if "sustainability_train.csv" in sust_files:
        sust_files.remove("sustainability_train.csv")

    # create a df to store predictions and another to store train
    df_pred = pd.DataFrame(
        columns=[
            "duration(s)",
            "power_consumption(kWh)",
            "kwh_scaled",
            "CO2_emissions(kg)",
            "co2_scaled",
        ]
    )
    df_train = pd.DataFrame(
        columns=[
            "duration(s)",
            "power_consumption(kWh)",
            "kwh_scaled",
            "CO2_emissions(kg)",
            "co2_scaled",
        ]
    )

    for file in sust_files:
        df = pd.read_csv(os.path.join(sustainability_path, file))

        # extract data for pred
        co2 = df.loc[0, "CO2_emissions(kg)"]
        power = df.loc[0, "power_consumption(kWh)"]
        co2_scaled = co2_transform(co2)
        kwh_scaled = co2_transform(power, mode="kwh")
        time = df.loc[0, "duration(s)"]

        # add data to df_pred as a new row
        df_pred.loc[file[:-4]] = [time, power, kwh_scaled, co2, co2_scaled]

        # extract data for train
        co2 = df.loc[2, "CO2_emissions(kg)"]
        power = df.loc[2, "power_consumption(kWh)"]
        co2_scaled = co2_transform(co2)
        kwh_scaled = co2_transform(power, mode="kwh")
        time = df.loc[2, "duration(s)"]

        # add data to df_train as a new row
        df_train.loc[file[:-4]] = [time, power, kwh_scaled, co2, co2_scaled]

    # round all values to 2 decimals and save
    df_pred = df_pred.round(2)
    df_pred.to_csv(os.path.join(sustainability_path, "sustainability_prediction.csv"))

    df_train = df_train.round(2)
    df_train.to_csv(os.path.join(sustainability_path, "sustainability_train.csv"))


@click.command()
@click.option("--results_folders", "-r", type=str, multiple=True)
@click.option("--path", type=str, default="results")
@click.option(
    "--chemistry", type=bool, default=True, help="Whether to compute chemistry metrics or not."
)
@click.option(
    "--mapping",
    type=bool,
    default=False,
    help="Whether to compute mapping and templates or not (these are required for chemistry metrics).",
)
def main(results_folders, path, chemistry, mapping):
    """Main results analysis pipeline for the metrics.
    Args:
    results_folders: Path. If previous results exist, load them from this path
    path: Compute results and store them in this path
    chemistry: Bool. whether the model is a chemistry model or not
    mapping: str. mapping to use to compute the metrics
    """

    if results_folders:
        print("Extracting results from folders...")
        extract_results(results_folders)

    if path:
        print("Computing results...")
        compute_results(path, chemistry, mapping)


if __name__ == "__main__":
    main()
