# Author: Tim Golombeck tjg2075

import json
import math
import os
import pandas as pd
import random
import re
import sys

class ReviewDecisionTreeNode:

    # Initializes a Binary Decision Tree Node.
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # Used to get the weighted size of the Child Nodes.
    def weighted_child_size(self, ratings, class_weights):
        return sum(class_weights.get(r, 1.0) for r in ratings)

    # Used to calculate the Weighted Gini Impurity at each threshold to determine the best split.
    def weighted_gini(self, ratings, class_weights):
        # Count the number of appearances of each rating.
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1

        # Calculate the Weighted Total.
        weighted_total = 0
        for rating in rating_counts:
            weighted_total += class_weights.get(rating, 1.0) * rating_counts[rating]

        # Calculate Weighted Gini Impurity (1 - Summation weights*probabilites squared).
        weighted_gini = 1
        for rating in rating_counts:
            weight = class_weights.get(rating, 1.0)
            probability = (weight * rating_counts[rating])/ weighted_total
            weighted_gini -= probability ** 2

        return weighted_gini

    # Calculate the Gini Gain using Weighted Ginis.
    def gini_gain(self, parent, left, right, class_weights):
        # Calculate the Ginis of the parent node and the left/right splits.
        parent_gini = self.weighted_gini(parent, class_weights)
        left_child_gini = self.weighted_gini(left, class_weights)
        right_child_gini = self.weighted_gini(right, class_weights)

        # Use the Weighted Sizes to calculate Child Gini.
        left_weight = self.weighted_child_size(left, class_weights)
        right_weight = self.weighted_child_size(right, class_weights)
        total_weight = left_weight + right_weight

        # Calculate the weighted gini of the child splits.
        weighted_child_gini = (left_weight/total_weight)*left_child_gini + (right_weight/total_weight)*right_child_gini

        # Return the Gini Gain.
        return parent_gini - weighted_child_gini

    # Used to calculate the Weighted Entropy at each threshold to determine the best split.
    def weighted_entropy(self, ratings, class_weights):
        # Count the number of appearances of each rating.
        rating_counts = {}
        for rating in ratings:
            if rating not in rating_counts:
                rating_counts[rating] = 0
            rating_counts[rating] += 1
        
        # Calculate the Weighted Total.
        weighted_total = 0
        for rating in rating_counts:
            weighted_total += class_weights.get(rating, 1.0) * rating_counts[rating]

        # Calculate Entropy (- Summation weight*probabilites log2 weight*probabilities).
        weighted_entropy = 0
        for rating in rating_counts:
            weight = class_weights.get(rating, 1.0)
            probability = (weight * rating_counts[rating]) / weighted_total
            weighted_entropy -= probability * math.log2(probability)
        
        return weighted_entropy
    
    # Calculate the Information Gain using Weighted Entropies.
    def information_gain(self, parent, left, right, class_weights):
        # Calculate the Entropies of the parent node and the left/right splits.
        parent_entropy = self.weighted_entropy(parent, class_weights)
        left_child_entropy = self.weighted_entropy(left, class_weights)
        right_child_entropy = self.weighted_entropy(right, class_weights)

        # Use the Weighted Sizes to calculate Child Entropy.
        left_weight = self.weighted_child_size(left, class_weights)
        right_weight = self.weighted_child_size(right, class_weights)
        total_weight = left_weight + right_weight

        # Calculate the weighted entropy of the child splits.
        weighted_child_entropy = (left_weight/total_weight)*left_child_entropy + (right_weight/total_weight)*right_child_entropy

        # Return the Information Gain.
        return parent_entropy - weighted_child_entropy
    
    # Used to determine the Majority Class in a Leaf Node.
    def weighted_majority(self, ratings, class_weights):
        scores = {}
        for r in ratings:
            scores[r] = scores.get(r, 0) + class_weights.get(r, 1.0)
        return max(scores, key=scores.get)

    # Calculate the Best Word and Threshold to split the Tree at.
    def best_split(self, feature_vectors, ratings, params, class_weights):
        # Initialize Best Word, Threshold and Gain.
        best_gain = -1
        best_word = None
        best_threshold = None

        # Collect all of the words to limit the number of features to check for the split.
        all_words = list(feature_vectors[0].keys()) if feature_vectors else []

        if params["max_features"] is not None:
            features = random.sample(all_words, min(params["max_features"], len(all_words)))
        else:
            features = all_words

        # Loop over each word in the potentially limited features to determine the best split.
        for word in features:
            # Binary split: word absent (0) vs present (1)
            left_ratings = [ratings[i] for i in range(len(feature_vectors))
                            if feature_vectors[i].get(word, 0) == 0]

            right_ratings = [ratings[i] for i in range(len(feature_vectors))
                            if feature_vectors[i].get(word, 0) == 1]

            # Skip any invalid splits.
            if len(left_ratings) == 0 or len(right_ratings) == 0:
                continue
                
            # Calculate the Gain for the split based on the criterion.
            if params["criterion"] == "gini":
                gain = self.gini_gain(ratings, left_ratings, right_ratings, class_weights)
            else:
                gain = self.information_gain(ratings, left_ratings, right_ratings, class_weights)

            if gain > best_gain:
                best_gain = gain
                best_word = word
                best_threshold = 0.5 # Since Binary Best threshold is always 0.5.

        return best_gain, best_word, best_threshold
    
    # This is used to build the entire DT using the Feature vectors and Ratings.
    def build_tree(self, feature_vectors, ratings, params, class_weights, depth=0):
        # If all ratings are the same, Return a new Leaf Node.
        if len(set(ratings)) == 1:
            return ReviewDecisionTreeNode(value=ratings[0])
        
        # If the depth is greater than or equal to the max depth
        # create a new Leaf Node.
        if depth >= params["max_depth"]:
            majority_rating = self.weighted_majority(ratings, class_weights)
            return ReviewDecisionTreeNode(value=majority_rating)
    
        # Stop if too few samples.
        if len(ratings) < params["min_samples_split"]:
            majority_rating = self.weighted_majority(ratings, class_weights)
            return ReviewDecisionTreeNode(value=majority_rating)

        # Find the Best Split.
        info_gain, word, threshold = self.best_split(feature_vectors, ratings, params, class_weights)

        # If no valuable information can be gained or no best word is found
        # create a new Leaf Node.
        if info_gain <= params["min_info_gain"] or word is None:
            majority_rating = self.weighted_majority(ratings, class_weights)
            return ReviewDecisionTreeNode(value=majority_rating)

        # Split the Data.
        left_feature_vectors = []
        right_feature_vectors = []
        left_ratings = []
        right_ratings = []

        for index, review_vector in enumerate(feature_vectors):
            value = review_vector.get(word, 0)
            if value <= threshold:
                left_feature_vectors.append(review_vector)
                left_ratings.append(ratings[index])
            else:
                right_feature_vectors.append(review_vector)
                right_ratings.append(ratings[index])

        # Check if the the leaf nodes would be too small after the split, if so create a new Leaf Node.
        if len(left_ratings) < params["min_samples_leaf"] or len(right_ratings) < params["min_samples_leaf"]:
            majority_rating = self.weighted_majority(ratings, class_weights)
            return ReviewDecisionTreeNode(value=majority_rating)

        # Create the Nodes.
        left_node = self.build_tree(left_feature_vectors, left_ratings, params, class_weights, depth + 1)
        right_node = self.build_tree(right_feature_vectors, right_ratings, params, class_weights, depth + 1)

        return ReviewDecisionTreeNode(feature=word, threshold=threshold, left=left_node, right=right_node)
    
# We convert the Node to a Dictionary so that we can save the model into a JSON.
def convert_node_to_json(node):
    if node.value is not None:
        return {"value": node.value}
    
    return {
        "feature": node.feature,
        "threshold": node.threshold,
        "left": convert_node_to_json(node.left),
        "right": convert_node_to_json(node.right)
    }

# Used to calculate the Class Weights to later Calculate the Weighted Entropy/Gini.
def compute_class_weights(all_ratings):
    class_counts = {}

    # Count the number of times each Rating appears.
    for rating in all_ratings:
        if rating not in class_counts:
            class_counts[rating] = 0
        class_counts[rating] += 1

    total = len(all_ratings)
    num_classes = len(class_counts)

    class_weights = {}

    # Calculate the Inverse Frequency of each class to use as the Weights (Sqrt for Softer Weights).
    for rating, rating_count in class_counts.items():
        average_num_ratings = total / num_classes
        class_weights[rating] = math.sqrt(average_num_ratings/ rating_count)

    return class_weights

def main():
    if len(sys.argv) != 3:
        print("Usage: python build_dt_model.py <path_to_training_feature_json> <path_to_training_data_csv>")
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

    # Initialize the hyperparamters for the Decision Tree (Using best Hyperparameters found in Evaluation).
    params = {
        "criterion": "entropy",
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "min_info_gain": 0,
        "max_features": None
    }

    dt = ReviewDecisionTreeNode()

    # Set random seed for reproducibility.
    random.seed(42)

    # Obtain the class weights to be used in training.
    class_weights = compute_class_weights(train_target)

    # After all of the files are loaded, create the DT.
    root = dt.build_tree(feature_train, train_target, params, class_weights)

    # Convert the DT to a dictionary and then save it to a JSON file.
    tree_json = convert_node_to_json(root)

    base_dir = os.path.dirname(json_file_path)
    filename = os.path.basename(json_file_path)
    data_dir = os.path.dirname(base_dir)
    dt_dir = os.path.join(data_dir, "trained_dt_models")

    # Extract the random state number used in the training Feature filename.
    match = re.search(r'(\d+)\.json', filename)
    random_state = match.group(1) if match else "0"

    # Construct the output path.
    trained_dt_path = os.path.join(dt_dir, f"trained_dt_model{random_state}.json")

    # Save the trained DT JSON.
    with open(trained_dt_path, "w") as file:
        json.dump(tree_json, file)

    print(f"Trained DT save to: {trained_dt_path}")

if __name__ == "__main__":
    main()