# Author: Tim Golombeck tjg2075

import json
import os
import pandas as pd
import random
import sys
from build_dt_model import ReviewDecisionTreeNode, compute_class_weights, convert_node_to_json

class RandomForest:

    # Initializes a Random Forest Object.
    def __init__(self, number_trees, params):
        self.number_trees = number_trees
        self.params = params
        self.decision_trees = []

    # Creates a Bootstrap Sample of the dataset by Sampling
    # with replacement from the training data.
    def bootstrap_sample(self, feature_vectors, ratings):

        # The size of the features.
        feature_size = len(feature_vectors)

        # Samples of the indices to bootstrap features and ratings.
        sampled_indices = [
            random.randint(0, feature_size - 1) for _ in range(feature_size)
        ]  

        # The Bootstrapped feature_vectors and ratings based on the sampled indices.
        bootstrap_feature_vectors = [
            feature_vectors[index] for index in sampled_indices
        ] 

        bootstrap_ratings = [
            ratings[index] for index in sampled_indices
        ]

        return bootstrap_feature_vectors, bootstrap_ratings
    
    # Builds Multiple DTs using bootstrapping samples of the data.
    def fit(self, feature_vectors, ratings, class_weights):

        self.decision_trees = []

        # Train each tree in the Forest independently.
        for tree_index in range(self.number_trees):
            print(f"Training Tree {tree_index + 1}/{self.number_trees}")

            # Create bootstrap sample for the tree.
            bootstrap_feature_vectors, bootstrap_ratings = self.bootstrap_sample(feature_vectors, ratings)

            # Create a new DT instance.
            dt = ReviewDecisionTreeNode()

            # Train the DT using the bootstrapped sample.
            root = dt.build_tree(bootstrap_feature_vectors, bootstrap_ratings, self.params, class_weights) 

            # Store the trained DT into the forest.
            self.decision_trees.append(root)

# We convert the forest into a Dictionary so we can save it to a JSON.
def convert_forest_to_json(forest):
    return {
        "model_type": "random_forest",
        "number_trees": forest.number_trees,
        "params": forest.params,
        "trees": [
            {
                "tree_id": tree_index,
                "tree": convert_node_to_json(tree_root)
            }
            for tree_index, tree_root in enumerate(forest.decision_trees)
        ]
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python build_random_forest.py <path_to_training_feature_json> <path_to_training_data_csv>")
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
    
    ratings = all_data['star_rating'].tolist()

    # Obtain the class weights.
    class_weights = compute_class_weights(ratings)

    # Use the hyperparamters found to be the best for a single DT.
    params = {
        "criterion": "entropy",
        "max_depth": 25,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "min_info_gain": 0,
        "max_features": None
    }

    # Train the random forest.
    random.seed(42)

    forest = RandomForest(number_trees=200, params=params)

    forest.fit(feature_train, ratings, class_weights)

    # Save the model to a JSON.
    forest_json = convert_forest_to_json(forest)

    output_path = os.path.join(os.path.dirname(os.path.dirname(json_file_path)), "trained_dt_models/random_forest_model42.json")

    with open(output_path, "w") as file:
        json.dump(forest_json, file)

    print(f"Saved Random Forest to: {output_path}")

if __name__ == "__main__":
    main()