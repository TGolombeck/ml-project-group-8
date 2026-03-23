# Author: Tim Golombeck tjg2075

import pandas as pd

# This is the main method used to combine all of the research data used in this project.
# It first create pandas DataFrames of the data, renames the columns so that the data can be 
# combined properly then added into a new csv file called all_data.csv.
def main():
    
    # The First Data .csv file has an unnamed column at index 0 to hold the ID, we do not need the ID
    # so we drop it when adding the data to a DataFrame.
    ratingsdf1 = pd.read_csv("../data/Rating_Prediction_dataset.csv").drop(columns=["Unnamed: 0"])
    ratingsdf2 = pd.read_csv("../data/Ratings.csv")

    # For the data to combine properly the name of the columns also must be the same.
    ratingsdf1 = ratingsdf1.rename(columns={
        "Product_Review": "review",
        "Ratings": "star_rating"
    })

    ratingsdf2 = ratingsdf2.rename(columns={
        "reviews": "review",
        "rating": "star_rating"
    })

    # We combine the two DataFrames here ignoring any indexes.
    all_data = pd.concat([ratingsdf1, ratingsdf2], ignore_index=True)

    # We ignore all rows with NaN/Null/None values.
    all_data = all_data.dropna()

    # We then add that DataFrame to a new file to be used in our Models.
    all_data.to_csv("../data/all_data.csv", index=False)

    # After, a success message is displayed to confirm that all_data.csv was created.
    print("CSV files combined!")

    
if __name__ == "__main__":
    main()