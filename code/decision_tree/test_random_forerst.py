# Author: Tim Golombeck tjg2075

import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import sys

# Traverse a Single Tree and Return a Prediction.
def predict_tree(node, sample):
    # If Leaf Node then return prediction.
    if "value" in node:
        return node["value"]
    
    feature = node["feature"]
    threshold = node["threshold"]

    value = sample.get(feature, 0)

    if value <= threshold:
        return predict_tree(node["left"], sample)
    else:
        return predict_tree(node["right"], sample)
    
# Traverse the Forest and return the majority prediction.
def predict_forest_label(forest, sample):
    tree_predictions = []

    for tree in forest["trees"]:
        tree_predictions.append(predict_tree(tree["tree"], sample))

    return max(set(tree_predictions), key=tree_predictions.count)

# Calculates the  Forest Probability Score for ROC curve (Binary Values).
def forest_score(forest, sample):
    tree_predictions = [predict_tree(tree["tree"], sample) for tree in forest["trees"]]

    return sum(1 if pred >= 5 else 0 for pred in tree_predictions) / len(tree_predictions)

# Converts to a Binary Value for TPR and FPR for metrics.
def convert_to_binary_rating(rating):
    if rating >= 5:
        return 1
    else:
        return 0

# Computes the True Positive Rate and False Positive Rate for the ROC Curve.
def compute_tpr_fpr(actual, predicted):
    # Computes the sum of true positives, false positives, true negatives, and false negatives.
    tp = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
    fn = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)

    # Compute the total TPR and FPR.
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return tpr, fpr

# Computes the AUC for the ROC Curve using the FPR and TPR values.
def compute_auc(fpr_list, tpr_list):
    # Sort the FPR and TPR values in ascending order of FPR.
    sorted_pairs = sorted(zip(fpr_list, tpr_list), key=lambda points: points[0])

    # Calculate the AUC by finding the Trapezoid Area.
    auc = 0.0

    for index in range(1, len(sorted_pairs)):
        fpr1, tpr1 = sorted_pairs[index - 1]
        fpr2, tpr2 = sorted_pairs[index]

        # Calculate the area of the trapezoid formed by the two points and add it to the AUC.
        auc += (fpr2 - fpr1) * (tpr1 + tpr2) / 2

    return auc

def main():
    if len(sys.argv) != 4:
        print("Usage: python test_random_forest.py <path_to_random_forest_json> <path_to_testing_feature_json> <path_to_testing_data_csv>")
        sys.exit(1)

    random_forest_path = sys.argv[1]
    feature_file_path = sys.argv[2]
    test_file_path = sys.argv[3]

    # Verify that the files exist.
    for path in [random_forest_path, feature_file_path, test_file_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    # Open Random Forest JSON.
    with open(random_forest_path, "r") as file:
        forest = json.load(file)

    # Open Testing Feature JSON.
    with open(feature_file_path, "r") as file:
        feature_test = json.load(file)

    # Load CSV data.
    test_data = pd.read_csv(test_file_path)
    actual_ratings = test_data["star_rating"].tolist()

    # Extract the random state number used in the testing Feature filename.
    filename = os.path.basename(feature_file_path)
    match = re.search(r'(\d+)\.json', filename)
    dataset_id = match.group(1) if match else "0"

    # Actual Binary Values.
    actual_binaries = [convert_to_binary_rating(rating) for rating in actual_ratings]

    # Prepare the Metrics.
    predicted_ratings = []
    predicted_binaries = []

    for index, sample in enumerate(feature_test):

        # Predict the Rating.
        prediction = predict_forest_label(forest, sample)

        predicted_ratings.append(prediction)

        # Convert Prediction to Binary.
        predicted_binary = convert_to_binary_rating(prediction)

        predicted_binaries.append(predicted_binary)

    # Calculate Accuracy of Forest.
    correct_predictions = sum(
        1 for actual, predicted in zip(actual_ratings, predicted_ratings)
        if actual == predicted
    )
    accuracy = correct_predictions / len(actual_ratings)

    # Calculate Binary Accuracy of Forest.
    correct_binary =  sum(
        1 for actual, predicted in zip(actual_binaries, predicted_binaries)
        if actual == predicted
    )
    binary_accuracy = correct_binary / len(actual_binaries)

    # Compute the Scores for the Forest.
    forest_probability_scores = [
        forest_score(forest, sample)
        for sample in feature_test
    ]

    # ROC and AUC using Forest Probability Score.
    # Sort thresholds from actual observed scores
    thresholds = sorted(set(forest_probability_scores))

    roc_points_fpr = []
    roc_points_tpr = []

    # For each threshold based on the number of trees.
    for threshold in thresholds:
        
        # Compute the threshold predictions.
        threshold_prediction = [
            1 if prob_score >= threshold else 0
            for prob_score in forest_probability_scores
        ]

        # Calculate tpr and fpr based on give tree threshold.
        tpr, fpr = compute_tpr_fpr(actual_binaries, threshold_prediction)

        roc_points_fpr.append(fpr)
        roc_points_tpr.append(tpr)

    # Create ROC curve.
    sorted_roc = sorted(zip(roc_points_fpr, roc_points_tpr))
    fpr_sorted, tpr_sorted = zip(*sorted_roc)

    # Add (0,0) and (1,1) to make the Curve look better.
    fpr_sorted = [0.0] + list(fpr_sorted) + [1.0]
    tpr_sorted = [0.0] + list(tpr_sorted) + [1.0]

    plt.figure()
    plt.plot(fpr_sorted, tpr_sorted, marker='o', color='blue', label=f"ROC Random Forest Model")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Coin-Flip Line')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.margins(0)
    plt.grid(True)

    # Save the ROC curve.
    # Get the output directory to store results.
    output_dir = os.path.dirname(feature_file_path)

    roc_plot_path = os.path.join(output_dir, f"random_forest_roc_curve_{dataset_id}.png")
    plt.savefig(roc_plot_path)
    plt.close()

    # Calculate the AUC.
    auc = compute_auc(roc_points_fpr, roc_points_tpr)

    # Output the Metrics.
    print("\n================ RESULTS ================")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Binary Accuracy: {binary_accuracy:.4f}")
    print(f"TPR (last):      {roc_points_tpr[-1]:.4f}")
    print(f"FPR (last):      {roc_points_fpr[-1]:.4f}")
    print(f"AUC:             {auc:.4f}")
    print("========================================\n")

    print("Avg score positive class:",
      sum(forest_score(forest, s) for i, s in enumerate(feature_test)
          if actual_binaries[i] == 1) / sum(actual_binaries))

    print("Avg score negative class:",
        sum(forest_score(forest, s) for i, s in enumerate(feature_test)
            if actual_binaries[i] == 0) / (len(actual_binaries) - sum(actual_binaries)))
        
if __name__ == "__main__":
    main()