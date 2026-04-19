# Author: Tim Golombeck tjg2075

import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import sys
from test_dt_model import predict

# Generalizes the Star Ratings into Binary Classes for the purpose of calculating the TPR and FPR for the ROC Curve.
def generalize_star_rating(star_rating):
    if star_rating >= 5:
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
        print("Usage: python test_many_dt_models.py <path_to_trained_dt_models_dir> <path_to_testing_feature_json> <path_to_testing_data_csv>")
        sys.exit(1)

    dt_file_dir = sys.argv[1]
    feature_file_path = sys.argv[2]
    test_file_path = sys.argv[3]

    # Verify that the files exist.
    for path in [dt_file_dir, feature_file_path, test_file_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    # Open Testing Feature JSON.
    with open(feature_file_path, "r") as file:
        feature_test = json.load(file)

    # Load CSV data.
    test_data = pd.read_csv(test_file_path)

    # Extract the random state number used in the testing Feature filename.
    filename = os.path.basename(feature_file_path)
    match = re.search(r'(\d+)\.json', filename)
    dataset_id = match.group(1) if match else "0"

    # Loop over each DT model in the directory and store them in a list.
    dt_names = []
    dt_models = []
    for file in os.listdir(dt_file_dir):
        match = re.search(f'_{dataset_id}.json', file)
        if match:
            with open(os.path.join(dt_file_dir, file), "r") as f:
                dt_names.append(file)
                dt_models.append(json.load(f))

    # Load the actual star ratings and generalized binary classes for the test data.
    actual_star_ratings = test_data['star_rating'].tolist()
    actual_binary_classes = [generalize_star_rating(rating) for rating in actual_star_ratings]

    # Make predictions for each model and evaluate the results.
    results = {}

    points = []

    best_model = None
    best_accuracy = -1
    best_generalized_model = None
    best_generalized_accuracy = -1
    best_roc_model = None
    best_roc_score = -1

    # The main loop that iterates over each model to make predictions and evaluate the results.
    for model, name in zip(dt_models, dt_names):

        star_predictions = [predict(model, vector) for vector in feature_test]
        generalized_predictions = [generalize_star_rating(pred) for pred in star_predictions]

        # Calculate the Accuracy for the Star Rating Predictions before generalizaiton.
        star_accuracy = sum(
            1 for index in range(len(star_predictions)) 
            if star_predictions[index] == actual_star_ratings[index]
            ) / len(star_predictions)

        # Record the Best Model for Star Rating Predictions.
        if star_accuracy > best_accuracy:
            best_model = name
            best_accuracy = star_accuracy


        # Calculate the Accuracy for the Generalized Binary Class Predictions.
        generalized_accuracy = sum(
            1 for index in range(len(generalized_predictions)) 
            if generalized_predictions[index] == actual_binary_classes[index]
            ) / len(generalized_predictions)
        
        # Record the Best Model for Generalized Binary Class Predictions.
        if generalized_accuracy > best_generalized_accuracy:
            best_generalized_model = name
            best_generalized_accuracy = generalized_accuracy

        # Compute the TPR and FPR for the Generalized Binary Class Predictions.
        tpr, fpr = compute_tpr_fpr(actual_binary_classes, generalized_predictions)
        points.append((fpr,tpr))

        # Check if this model has the best ROC AUC score.
        roc_score = tpr - fpr  # A simple way to evaluate the ROC performance.
        if roc_score > best_roc_score:
            best_roc_model = name
            best_roc_score = roc_score

        print("Model:", name)

        print("Predicted binary distribution:")
        print("0s:", generalized_predictions.count(0))
        print("1s:", generalized_predictions.count(1))

        print("Actual binary distribution:")
        print("0s:", actual_binary_classes.count(0))
        print("1s:", actual_binary_classes.count(1))

        print("-" * 40)

        # Store the results for the model in the results dictionary.
        results[name] = {
            "star_accuracy": star_accuracy,
            "generalized_accuracy": generalized_accuracy,
            "tpr": tpr,
            "fpr": fpr
        }
    
    # Store the results for each model in a DataFrame.
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index.name = "model"
    results_df = results_df.reset_index()

    # Get the output directory to store results.
    output_dir = os.path.dirname(feature_file_path)

    # Plot the ROC Curve.
    plt.figure()

    # ROC Curve for the Models.
    points = list(set(points))
    points = sorted(points, key=lambda x: x[0])
    fpr_list, tpr_list = zip(*points)
    
    # Add (0,0) and (1,1) to make the Curve look better.
    fpr_list = [0.0] + list(fpr_list) + [1.0]
    tpr_list = [0.0] + list(tpr_list) + [1.0]

    plt.plot(fpr_list, tpr_list, marker='o', color='blue', label='DT Models')

    # Coin Flip Line for Reference.
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Coin-Flip Line')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.margins(0)

    plt.title('ROC Curve for Decision Tree Models')
    plt.legend()

    # Save the ROC Curve plot.
    roc_plot_path = os.path.join(output_dir, f"dt_roc_curve_{dataset_id}.png")
    plt.savefig(roc_plot_path)
    plt.close()

    # Save the results DataFrame to a CSV file.
    results_csv_path = os.path.join(output_dir, f"dt_hyperparameter_comparison_{dataset_id}.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Print the Results summary.
    print("\n================ RESULTS SUMMARY ================\n")

    print(f"Best Accuracy Model: {best_model}")
    print(f"Best Accuracy: {best_accuracy:.4f}\n")

    print(f"Best Generalized Model: {best_generalized_model}")
    print(f"Best Generalized Accuracy: {best_generalized_accuracy:.4f}\n")

    print(f"Best ROC Model: {best_roc_model}")
    print(f"Best ROC Score (TPR - FPR): {best_roc_score:.4f}\n")

    print(f"ROC AUC Score: {compute_auc(fpr_list, tpr_list):.4f}\n")

    print("Top 5 Models by Generalized Accuracy:\n")

    top5 = results_df.sort_values(by="generalized_accuracy", ascending=False).head(5)

    print(top5[["model", "generalized_accuracy", "tpr", "fpr"]])

    print(f"Total Models Evaluated: {len(dt_models)}")
    print("=================================================\n")

    print(f"ROC Curve saved to: {roc_plot_path}")
    print(f"Results CSV saved to: {results_csv_path}")

if __name__ == "__main__":
    main()