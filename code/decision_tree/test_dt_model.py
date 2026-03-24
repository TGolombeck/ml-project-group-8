#Author: Tim Golombeck tjg2075

import json
import os
import pandas as pd
import sys

# Predict the Star Rating of a Review.
def predict(node_dict, tfidf_vector):
    # Base Case if the node has a value.
    if "value" in node_dict and node_dict["value"] is not None:
        return node_dict["value"]
    
    # Otherwise, Recurse until Base Case.
    feature = node_dict.get("feature")
    threshold = node_dict.get("threshold", 0)

    # Get the Value for the feature from the Review's TF-IDF Vector.
    feature_value = tfidf_vector.get(feature, 0)

    if feature_value <= threshold:
        return predict(node_dict["left"], tfidf_vector)
    else:
        return predict(node_dict["right"], tfidf_vector)

def main():
    if len(sys.argv) != 4:
        print("Usage: python test_dt_model.py <path_to_trained_dt_json> <path_to_testing_tfidf_json> <path_to_testing_data_csv>")
        sys.exit(1)

    dt_file_path = sys.argv[1]
    tfidf_file_path = sys.argv[2]
    test_file_path = sys.argv[3]

    # Verify that the files exist.
    for path in [dt_file_path, tfidf_file_path, test_file_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    # Open DT JSON.
    with open(dt_file_path, "r") as file:
        tree_json = json.load(file)

    # Open Testing TF-IDF JSON.
    with open(tfidf_file_path, "r") as file:
        tfidf_test = json.load(file)

    # Load CSV data.
    test_data = pd.read_csv(test_file_path)

    # Make predictions.
    predictions = [predict(tree_json, vector) for vector in tfidf_test]

    # Add Prediction Column.
    test_data['predicted_star_rating'] = predictions

    # Calculate the Accuracy.
    correct = sum(1 for index in range(len(predictions)) if predictions[index] == test_data['star_rating'].iloc[index])
    accuracy = correct / len(predictions)
    print(f"Accuracy on Test Data: {accuracy*100:.2f}%")

    # Save predictions CSV.
    base_dir = os.path.dirname(test_file_path)
    filename = os.path.splitext(os.path.basename(test_file_path))[0]
    predictions_path = os.path.join(base_dir, f"predictions_{filename}.csv")

    test_data.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()