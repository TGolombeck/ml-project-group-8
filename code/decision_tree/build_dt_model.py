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

    # Used to calculate the Gini Impurity at each threshold to determine the best split.
    def gini(self, ratings):
        # The total ratings within a review.
        total_ratings = len(ratings)
        
        # Count the number of appearances of each rating.
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1

        # Calculate Gini Impurity (1 - Summation probabilites squared).
        gini = 1
        for count in rating_counts.values():
            probability = count / total_ratings
            gini -= probability ** 2

        return gini

    # Calculate the Gini Gain using Weighted Ginis.
    def gini_gain(self, parent, left, right):
        # The total number of ratings in the parent node.
        total_ratings = len(parent)

        # Calculate the Ginis of the parent node and the left/right splits.
        parent_gini = self.gini(parent)
        left_child_gini = self.gini(left)
        right_child_gini = self.gini(right)

        # Calculate the weighted gini of the child splits.
        weighted_child_gini = (len(left)/total_ratings)*left_child_gini + (len(right)/total_ratings)*right_child_gini

        # Return the Gini Gain.
        return parent_gini - weighted_child_gini

    # Used to calculate the Entropy at each threshold to determine the best split.
    def entropy(self, ratings):
        # The total ratings within a review.
        total_ratings = len(ratings)

        # Count the number of appearances of each rating.
        rating_counts = {}
        for rating in ratings:
            if rating not in rating_counts:
                rating_counts[rating] = 0
            rating_counts[rating] += 1
        
        # Calculate Entropy (- Summation probabilites log2 probabilities).
        entropy = 0
        for count in rating_counts.values():
            probability = count / total_ratings
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    # Calculate the Information Gain using Weighted Entropies.
    def information_gain(self, parent, left, right):
        # The total number of ratings in the parent node.
        total_ratings = len(parent)

        # Calculate the Entropies of the parent node and the left/right splits.
        parent_entropy = self.entropy(parent)
        left_child_entropy = self.entropy(left)
        right_child_entropy = self.entropy(right)

        # Calculate the weighted entropy of the child splits.
        weighted_child_entropy = (len(left)/total_ratings)*left_child_entropy + (len(right)/total_ratings)*right_child_entropy

        # Return the Information Gain.
        return parent_entropy - weighted_child_entropy
    
    # Calculate the Best Word and Threshold to split the Tree at.
    def best_split(self, tfidf_vectors, ratings, params):
        # Initialize Best Word, Threshold and Gain.
        best_gain = -1
        best_word = None
        best_threshold = None

        # Collect all of the words to limit the number of features to check for the split.
        all_words = list(tfidf_vectors[0].keys())

        if params["max_features"] is not None:
            max_features = min(params["max_features"], len(all_words))
            features = random.sample(all_words, min(params["max_features"], max_features))
        else:
            features = all_words

        # Loop over each word in the potentially limited features to determine the best split.
        for word in features:

            # Obtain a list of all tfidf values for the word.
            word_values = [review_vector.get(word, 0) for review_vector in tfidf_vectors]
            
            # Create a set that contains every TF-IDF value (thresholds) to check for the word.
            thresholds = list(set(word_values))

            # Limit the number of thresholds to check for the word to max_thresholds.
            thresholds = thresholds[:params["max_thresholds"]]

            for threshold in thresholds:
                # Create lists of left/right ratings for each threshold.
                left_ratings = [ratings[index] for index in range(len(tfidf_vectors))
                                if tfidf_vectors[index].get(word, 0) <= threshold]
                
                right_ratings = [ratings[index] for index in range(len(tfidf_vectors))
                                if tfidf_vectors[index].get(word, 0) > threshold]
                
                # Skip any invalid splits.
                if len(left_ratings) == 0 or len(right_ratings) == 0:
                    continue
                
                # Calculate the Gain for the split based on the criterion.
                if params["criterion"] == "gini":
                    gain = self.gini_gain(ratings, left_ratings, right_ratings)
                else:
                    gain = self.information_gain(ratings, left_ratings, right_ratings)

                if gain > best_gain:
                    best_gain = gain
                    best_word = word
                    best_threshold = threshold

        return best_gain, best_word, best_threshold
    
    # This is used to build the entire DT using the TF-IDF vectors and Ratings.
    def build_tree(self, tfidf_vectors, ratings, params, depth=0):
        # If all ratings are the same, Return a new Leaf Node.
        if len(set(ratings)) == 1:
            return ReviewDecisionTreeNode(value=ratings[0])
        
        # If the depth is greater than or equal to the max depth
        # create a new Leaf Node.
        if depth >= params["max_depth"]:
            majority_rating = max(set(ratings), key=ratings.count)
            return ReviewDecisionTreeNode(value=majority_rating)
    
        # Stop if too few samples.
        if len(ratings) < params["min_samples_split"]:
            majority_rating = max(set(ratings), key=ratings.count)
            return ReviewDecisionTreeNode(value=majority_rating)

        # Find the Best Split.
        info_gain, word, threshold = self.best_split(tfidf_vectors, ratings, params)

        # If no valuable information can be gained or no best word is found
        # create a new Leaf Node.
        if info_gain <= params["min_info_gain"] or word is None:
            majority_rating = max(set(ratings), key=ratings.count)
            return ReviewDecisionTreeNode(value=majority_rating)

        # Split the Data.
        left_tfidf_vectors = []
        right_tfidf_vectors = []
        left_ratings = []
        right_ratings = []

        for index, review_vector in enumerate(tfidf_vectors):
            value = review_vector.get(word, 0)
            if value <= threshold:
                left_tfidf_vectors.append(review_vector)
                left_ratings.append(ratings[index])
            else:
                right_tfidf_vectors.append(review_vector)
                right_ratings.append(ratings[index])

        # Check if the the leaf nodes would be too small after the split, if so create a new Leaf Node.
        if len(left_ratings) < params["min_samples_leaf"] or len(right_ratings) < params["min_samples_leaf"]:
            majority_rating = max(set(ratings), key=ratings.count)
            return ReviewDecisionTreeNode(value=majority_rating)

        # Create the Nodes.
        left_node = self.build_tree(left_tfidf_vectors, left_ratings, params, depth + 1)
        right_node = self.build_tree(right_tfidf_vectors, right_ratings, params, depth + 1)

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

def main():
    if len(sys.argv) != 3:
        print("Usage: python build_dt_model.py <path_to_training_tfidf_json> <path_to_training_data_csv>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    # Verify Inputed JSON file exists.
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        sys.exit(1)

    with open(json_file_path, "r") as file:
        tfidf_train = json.load(file)

    csv_file_path = sys.argv[2]

    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        sys.exit(1)

    # Read the data from the CSV file.
    all_data = pd.read_csv(csv_file_path)

    train_target = all_data['star_rating'].tolist()

    # Initialize the hyperparamters for the Decision Tree.
    params = {
        "criterion": "entropy",
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "min_info_gain": 1e-3,
        "max_features": 100,
        "max_thresholds": 20
    }

    dt = ReviewDecisionTreeNode()

    # Set random seed for reproducibility.
    random.seed(42)

    # After all of the files are loaded, create the DT,
    root = dt.build_tree(tfidf_train, train_target, params)

    # Convert the DT to a dictionary and then save it to a JSON file.
    tree_json = convert_node_to_json(root)

    base_dir = os.path.dirname(json_file_path)
    filename = os.path.basename(json_file_path)
    data_dir = os.path.dirname(base_dir)
    dt_dir = os.path.join(data_dir, "trained_dt_models")

    # Extract the random state number used in the training TF-IDF filename.
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