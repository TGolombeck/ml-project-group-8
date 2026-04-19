# Author: Tim Golombeck tjg2075

from itertools import product
import json
import os
import pandas as pd
import random
import re
import sys
from build_dt_model import ReviewDecisionTreeNode
from build_dt_model import compute_class_weights
from build_dt_model import convert_node_to_json

def main():
    if len(sys.argv) != 3:
        print("Usage: python build_many_dt_models.py <path_to_training_feature_json> <path_to_training_data_csv>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    # Verify Inputed JSON file exists.
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        sys.exit(1)

    with open(json_file_path, "r") as file:
        feature_train = json.load(file)

    csv_file_path = sys.argv[2]

    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        sys.exit(1)

    # Read the data from the CSV file.
    all_data = pd.read_csv(csv_file_path)

    train_target = all_data['star_rating'].tolist()

    # Set output directory for the models.
    base_dir = os.path.dirname(json_file_path)
    data_dir = os.path.dirname(base_dir)
    output_dir = os.path.join(data_dir, "trained_dt_models")

    # Create the hyperparameters to use for building the models.
    criterion = ["gini", "entropy"]
    max_depth = [5, 10, 15]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 5]
    min_info_gain = [0, 1e-4]
    max_features = [200, 500, None]

    param_grid = []

    # Add all combinations of hyperparameters to the param_grid list.
    for c, md, mss, msl, mig, mf in product(
        criterion, 
        max_depth, 
        min_samples_split, 
        min_samples_leaf, 
        min_info_gain, 
        max_features):

        param_grid.append({
            "criterion": c,
            "max_depth": md,
            "min_samples_split": mss,
            "min_samples_leaf": msl,
            "min_info_gain": mig,
            "max_features": mf,
        })

    # Create a ReviewDecisionTreeNode instance to build the models. 
    dt = ReviewDecisionTreeNode()

    # Extract the random state number used in the training Feature filename.
    filename = os.path.basename(json_file_path)
    match = re.search(r'(\d+)\.json', filename)
    dataset_id = match.group(1) if match else "0"

    # Obtain the class weights to be used in training.
    class_weights = compute_class_weights(train_target)

    # Loop over each combination of hyperparameters to build a model and save it as a JSON file.
    for index, params in enumerate(param_grid):
        print(f"Building model {index+1}/{len(param_grid)} with parameters: {params}")

        # Set random seed for reproducibility per Tree.
        random.seed(42 + index)

        # Build the Decision Tree using the current set of hyperparameters and convert it to JSON format.
        root = dt.build_tree(feature_train, train_target, params, class_weights)

        # Convert the tree to a JSON-serializable format.
        tree_json = convert_node_to_json(root)

        # Encode the hyperparameters into the filename for the model.
        crit = "Gi" if params["criterion"] == "gini" else "En"

        mf = "None" if params["max_features"] is None else str(params["max_features"])

        filename = (
            f"dt_{index + 1}_{crit}"
            f"_MD{params['max_depth']}"
            f"_MSS{params['min_samples_split']}"
            f"_MSL{params['min_samples_leaf']}"
            f"_MIG{params['min_info_gain']}"
            f"_MF{mf}"
            f"_{dataset_id}.json"
        )

        save_path = os.path.join(output_dir, filename)

        # Save the model as a JSON file.
        with open(save_path, "w") as file:
            json.dump(tree_json, file)

if __name__ == "__main__":
    main()