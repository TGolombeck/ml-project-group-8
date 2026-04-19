# Author: Tim Golombeck tjg2075

import json
import math
import os
import pandas as pd
import re
import sys

# This is used to clean the text from all of the reviews.
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove all/any emojis in the text
    # This works because emojis don't have an ASCII representation, so encoding
    # the text into ascii while ignoring all characters that cannot will remove 
    # all emojis.
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove any newline or tab characters
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Replace any punctuation and special characters with ''
    # We use python's built in regualar expression library to do so (re)
    # What this is doing is it is taking a string literal, and replacing any character
    # that is not a word or a space with a space character.
    text = re.sub(r'[^\w\s]', '', text)

    # Make sure all spaces are collasped into a single space.
    # "Hello          World" -> "Hello World" 
    text = re.sub(r'\s+', ' ', text).strip()

    # Make the text lowercase
    text = text.lower()

    return text

# This method calculates the Binary Feature Vector for one review.
def calculate_features(tokens, all_words):
    # Create a dictionary for the review that holds all words and binary state.
    feature_dict = {}

    # Calculate the total amount of words in the review.
    total_words = len(tokens)

    for word in all_words:
        if word in tokens:
            feature_dict[word] = 1
    
    return feature_dict

# Determines the stopwords that appear in almost every review.
def build_stopwords(all_tokens, all_words, threshold=0.7):
    # Get the number of tokens.
    num_tokens = len(all_tokens)

    # Initialize the stopword set.
    stopword = set()

    # Loop through all of the words, if a word appears in >=threshold amount of reviews,
    for word in all_words:
        count = sum(1 for tokens in all_tokens if word in tokens)

        if count / num_tokens > threshold:
            stopword.add(word)
    
    return stopword

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python dt_preprocessing.py <path_to_csv_file>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Verify Inputed CSV file exists.
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        sys.exit(1)

    # Read the data from the CSV file.
    all_data = pd.read_csv(csv_file_path)
    
    all_data['cleaned_review'] = all_data['review'].apply(clean_text)

    # Randomize the DataFrame before splitting into Train/Test Sets.
    random_state = 42
    all_data = all_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # The size of the Test Set in proportion to all of the data.
    test_size = 0.2

    # The size of all of all of the data. 
    total_reviews = len(all_data)
    
    # The index the data will be split at.
    split_index = int((1 - test_size) * total_reviews)

    # Now we split the data and target variables into their own DataFrames.
    train_data = all_data['cleaned_review'].iloc[:split_index].tolist()
    test_data = all_data['cleaned_review'].iloc[split_index:].tolist()

    train_target = all_data['star_rating'].iloc[:split_index].tolist()
    test_target = all_data['star_rating'].iloc[split_index:].tolist()

    # After we complete the Train/Test split, we start by tokenizing 
    # all of the words within the training reviews.
    train_tokens = [review.split() for review in train_data]

    # We will add all of words within train_tokens into a set, to automatically
    # remove any duplicate words.
    all_words = set()
    for token in train_tokens:
        all_words.update(token)

    # Sort the set alphabetically.
    all_words = sorted(list(all_words))

    # The frequency that a word appears in a review.
    doc_freq = {
        word: sum(1 for token in train_tokens if word in token)
        for word in all_words
    }

    # Remove any word that appears 2 or less times or appear too much.
    min_word_count = 5
    max_word_count = 0.8 * len(train_tokens)

    all_words = [
    word for word in all_words
    if doc_freq[word] >= min_word_count
    and doc_freq[word] <= max_word_count
    ]

    # Find and Remove Stopwords from all_words.
    stopwords = build_stopwords(train_tokens, all_words)

    all_words = [
        word for word in all_words 
        if word not in stopwords
    ]

    # Limit the amount of words in all_words.
    all_words = all_words[:2000]

    # Creates a list of training Dictionaries with the respective Binary Feature Vector for each review.
    feature_train = [calculate_features(tokens, all_words) for tokens in train_tokens]

    # We also need to tokenize the test data for the test set.
    test_tokens = [review.split() for review in test_data]

    # Creates a list of testing Dictionaries with the respective Binary Feature Vector for each review.
    feature_test = [calculate_features(tokens, all_words) for tokens in test_tokens]

    # Once the Feature Dictionaries are created for both Training and Testing Data,
    # we can save all of the information to different files to be used to build the 
    # DecisionTree Model and to test the DecisionTree Model.
    base_dir = os.path.dirname(csv_file_path)

    dt_dir = os.path.join(base_dir, "dt_data")

    train_data_path = os.path.join(dt_dir, f"dt_training_data{random_state}.csv")
    test_data_path  = os.path.join(dt_dir, f"dt_testing_data{random_state}.csv")
    train_feature_path = os.path.join(dt_dir, f"dt_training_feature{random_state}.json")
    test_feature_path  = os.path.join(dt_dir, f"dt_testing_feature{random_state}.json")

    # Save train/test CSVs.
    pd.DataFrame({"review": train_data, "star_rating": train_target}).to_csv(train_data_path, index=False)
    pd.DataFrame({"review": test_data, "star_rating": test_target}).to_csv(test_data_path, index=False)

    # Save train/test Feature JSONs.
    with open(train_feature_path, "w") as file:
        json.dump(feature_train, file)

    with open(test_feature_path, "w") as file:
        json.dump(feature_test, file)

if __name__ == "__main__":
    main()