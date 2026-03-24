# Author: Tim Golombeck tjg2075

import json
import math
import os
import pandas as pd
import re
import sys

# This is used to clean the text from all of the reviews before calculating TF-IDF.
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

    # Replace any punctuation and special characters with space
    # We use python's built in regualar expression library to do so (re)
    # What this is doing is it is taking a string literal, and replacing any character
    # that is not a word or a space with a space character.
    text = re.sub(r'[^\w\s]', ' ', text)

    # Make sure all spaces are collasped into a single space.
    # "Hello          World" -> "Hello World" 
    text = re.sub(r'\s+', ' ', text).strip()

    # Make the text lowercase
    text = text.lower()

    return text

# This method calculates the Term Frequency for one review.
def calculate_tf(tokens, all_words):
    # Create a dictionary for the review that holds all words and frequency within the review.
    tf_dict = {}

    # Calculate the total amount of words in the review.
    total_words = len(tokens)

    # Calculate the term frequency of each word within all_words and add it to the dictionary.
    for word in all_words:
        # If total_words is less than 0, set the tf_value to 0.
        tf_dict[word] = tokens.count(word) / total_words if total_words > 0 else 0

    
    return tf_dict

# This method calculates the Inverse Document Frequency.
def calculate_idf(all_tokens, all_words):
    # Create a dictionary for the review that holds all words and IDF.
    idf_dict = {}

    # Calculate the total amount of tokens.
    num_tokens = len(all_tokens)
    for word in all_words:
        
        # Sum together the number of reviews that contain the word.
        count = sum(1 for tokens in all_tokens if word in tokens)

        # Calculate Smooth IDF for the word.
        idf_dict[word] = math.log((num_tokens + 1) / (count + 1)) + 1

    return idf_dict

# Computes the TF-IDF for a single review.
def calculate_tfidf(tf_dict, idf_dict, threshold=0):
    # Create a dictionary for the review that holds all words and TF-IDF.
    tfidf_dict = {}

    for word in tf_dict:
        # Calculate the TF-IDF value.
        tfidf_value = tf_dict[word] * idf_dict[word]

        # Check to make sure that the value is above a threshold (0.01) for the Training Data.
        if tfidf_value >= threshold:
            tfidf_dict[word] = tfidf_value

    return tfidf_dict

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

    # Creates a list of training Dictionaries with the respective Term Frequency for each review.
    tf_train = [calculate_tf(tokens, all_words) for tokens in train_tokens]

    # Create a Dictionary of all words and Inverse Document Frequencies for the Training Data.
    idf_train = calculate_idf(train_tokens, all_words)

    # Calculates the Training TF-IDF Dictionaries for all reviews.
    tfidf_train = [calculate_tfidf(tf_dict, idf_train, threshold=0.01) for tf_dict in tf_train]

    # Once the TF-IDF Dictionaries are created for the Training Data, we do the same for the Testing Data.
    test_tokens = [review.split() for review in test_data]

    # Creates a list of Testing Dictionaries with the respective Term Frequency for each review.
    # We use all of the words in the TF-IDF for the Training Data so that no new words get added
    # and no old words get removed.
    tf_test = [calculate_tf(tokens, list(tfidf_train[0].keys())) for tokens in test_tokens]

    # Create the TF-IDF Dictionaries for the Testing Data using the IDF from the Training Data.
    tfidf_test = [calculate_tfidf(tf_dict, idf_train) for tf_dict in tf_test]

    # Once the TF-IDF Dictionaries are created for both Training and Testing Data,
    # we can save all of the information to different files to be used to build the 
    # DecisionTree Model and to test the DecisionTree Model.
    base_dir = os.path.dirname(csv_file_path)

    train_data_path = os.path.join(base_dir, f"dt_training_data{random_state}.csv")
    test_data_path  = os.path.join(base_dir, f"dt_testing_data{random_state}.csv")
    train_tfidf_path = os.path.join(base_dir, f"dt_training_tfidf{random_state}.json")
    test_tfidf_path  = os.path.join(base_dir, f"dt_testing_tfidf{random_state}.json")

    # Save train/test CSVs.
    pd.DataFrame({"review": train_data, "star_rating": train_target}).to_csv(train_data_path, index=False)
    pd.DataFrame({"review": test_data, "star_rating": test_target}).to_csv(test_data_path, index=False)

    # Save train/test TF-IDF JSONs.
    with open(train_tfidf_path, "w") as file:
        json.dump(tfidf_train, file)

    with open(test_tfidf_path, "w") as file:
        json.dump(tfidf_test, file)

if __name__ == "__main__":
    main()