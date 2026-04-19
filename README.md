# CSCI 335/635 Machine Learning Group Project – Group 8

## Abstract
This project implements multiple machine learning algorithms to address the real-world problem of predicting Amazon review star ratings from user reviews. The goal is to model the relationship between review features and numerical ratings in order to accurately infer sentiment and consumer feedback at scale. Several supervised learning approaches are trained and evaluated, including linear regression and logistic regression models, decision tree and random forest models, and transformer-based neural network models.

Each model is assessed based on its ability to capture patterns in review data and generalize to unseen examples. Performance is compared using standard evaluation metrics for both regression and classification tasks, providing insight into the trade-offs between model complexity, interpretability, and predictive accuracy. The results demonstrate how different machine learning paradigms perform on the same real-world sentiment prediction task, highlighting the advantages of ensemble and deep learning approaches in capturing nonlinear relationships in text-derived features.

---

## Developers
- Colin Cannell
- Tim Golombeck
- Carter Howell

---

# How to Run

## Data Preprocessing (data_csv.py)

### Purpose 

This script is responsible for preprocessing the raw Amazon review dataset and converting it into a structured CSV format suitable for the machine learning models. It extracts relevant review features, cleans and formats the data, and ensures consistency so that it can be reliably used in downstream models.

### How to Run 

To execute the preprocessing script, run the following command:

```bash
python data_csv.py
```

This script will combine 'Rating_Prediction_dataset.csv' and 'Ratings.csv' into 'all_data.csv' for use in the machine learning models.

## Decision Tree / Random Forest Models

### Purpose

These models are used to learn relationships between Amazon Reviews and their corresponding Star Ratings. Both approaches operate on the same feature-engineered dataset and are evaluated under the same experimental setup to ensure fair comparison. The Decision Tree Model serves as a Baseline Interpretale Classifier, while the Random Forest Model extends this approach by aggregating multiple Decision Trees to improve generalization and reduce overfitting. Together, they provide a comparison between single-model interpretability and ensemble-based performance improvements in predicting ratings based off of reviews.

### 1. Preprocessing (dt_preprocessing.py)

This script is responsible for cleaning, splitting, and tokenizing the Amazon review and rating data to make it interpretable for the Decision Tree and Random Forest models. It cleans and tokenizes the review text, extracts relevant features, and converts categorical and textual information into numerical representations that can be used by tree-based learning algorithms. The input is a CSV file containing the Amazon review and rating data, and the output consists of four files stored in /data/dt_data, including training and testing features and datasets used for model training and evaluation.

To execute the Decision Tree Preprocessing Script, run the following command:

```bash
python dt_preprocessing.py <path_to_csv_file>
```

Here is an example of the program being used in this project:

```bash
python dt_preprocessing.py ../../data/all_data.csv
```

The output will be saved to data/dt_data/ the files that are saved will be dt_training_features{random_state}.json, dt_training_data{random_state}.csv, dt_testing_features{random_state}.json, and dt_testing_data{random_state}.csv.

### 2. Single Decision Tree Model (build_dt_model.py, test_dt_model.py)

These scripts are responsible for training and evaluating a Decision Tree model. The trained model is stored as a JSON file in /data/trained_dt_models, and the prediction outputs generated during testing are saved in /data/dt_data.

To Build a Decision Tree Model, run the following command:

```bash
python build_dt_model.py <path_to_training_feature_json> <path_to_training_data_csv>
```

Here is an example of the program being used in this project:

```bash 
python build_dt_model.py ../../data/dt_data/dt_training_feature42.json ../../data/dt_data/dt_training_data42.csv
```

The output will be saved to data/train_dt_models, the file that is saved based on the example will be trained_dt_model42.json where 42 is the random state used to generate the train/test split.

To Test a Decision Tree Model, run the following command:

```bash
python test_dt_model.py <path_to_trained_dt_json> <path_to_testing_feature_json> <path_to_testing_data_csv>
```

Here is an example of the program being used in this project:

```bash
python test_dt_model.py ../../data/trained_dt_models/trained_dt_model42.json ../../data/dt_data/dt_testing_feature42.json ../../data/dt_data/dt_testing_data42.csv
```

The output will be saved to data/dt_data, the file that is saved based on the example will be predictions_dt_testing_data42.csv. The program will also print the testing accuracy of the model.

### 2. Multiple Decision Tree Model (build_many_dt_models.py, test_many_dt_models.py)

The purpose of these scripts is to perform hyperparameter tuning for the Decision Tree model in order to identify the best-performing configuration. build_many_dt_models.py generates and trains Decision Tree models across all combinations of selected hyperparameters, saving each trained model as a JSON file. test_many_dt_models.py then evaluates each model on the test dataset and records the corresponding performance metrics.

To Build the Multiple Decision Tree Models, run the following command:

```bash
python build_many_dt_models.py <path_to_training_feature_json> <path_to_training_data_csv>
```

Here is an example of the program being used in this project:

```bash
python build_many_dt_models.py ../../data/dt_data/dt_training_feature42.json ../../data/dt_data/dt_training_data42.csv
```

The output will be saved to data/trained_dt_models, the number of models saved will be based on the hyperparamter grid defined within the file.

To Test Multiple Decision Tree Models, run the following command:

```bash
python test_many_dt_models.py <path_to_trained_dt_models_dir> <path_to_testing_feature_json> <path_to_testing_data_csv>
```

Here is an example of the program being used in this project:

```bash
python test_many_dt_models.py ../../data/trained_dt_models ../../data/dt_data/dt_testing_feature42.json ../../data/dt_data/dt_testing_data42.csv
```

This program assumes that build_many_dt_models.py has already been ran. The output will be saved to data/dt_data, the files saved based on the example are dt_roc_curve_42.png which is an ROC curve of all of the hyperparamter models and dt_hyperparamter_comparison_42.csv which will contain the results of every hyperparameter. The program will also print information about each model as well as the top models found.

### 3. Random Forest Model (build_random_forest.py, test_random_forest.py)

The purpose of these scripts is to train and evaluate a Random Forest model for predicting Amazon review star ratings. build_random_forest.py constructs an ensemble of Decision Trees using the training dataset and saves the resulting model as a JSON file. test_random_forest.py then applies the trained Random Forest model to the test dataset and generates predictions, along with performance metrics for evaluation.

To Build a Random Forest Model, run the following command:

```bash
python build_random_forest.py <path_to_training_feature_json> <path_to_training_data_csv>
```

Here is an example of the program being used in this project:

```bash
python build_random_forest.py ../../data/dt_data/dt_training_feature42.json ../../data/dt_data/dt_training_data42.csv
```

The output will be saved to data/train_dt_models, the file that is saved based on the example will be random_forest_model42.json where 42 is the random state used to generate the train/test split.

To Test a Random Forest Model, run the following command:

```bash
python test_random_forest.py <path_to_random_forest_json> <path_to_testing_feature_json> <path_to_testing_data_csv>
```

Here is an example of the program being used in this project:

```bash
python test_random_forerst.py ../../data/trained_dt_models/random_forest_model42.json ../../data/dt_data/dt_testing_feature42.json ../../data/dt_data/dt_testing_data42.csv
```

The output will be saved to data/dt_data, the file that is saved based on the example will be random_forest_roc_curve_42.png which will contain the ROC curve image for the Random Forest Model. This program will also print the metrics of the Random Forest Model on the Testing Set.

## Linear / Logistic Regression Models

## Transformer Model
