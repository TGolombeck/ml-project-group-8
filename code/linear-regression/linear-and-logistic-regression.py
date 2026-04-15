import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ratings = []
reviews = []
with open("../../data/Ratings.csv", "r") as file:
    next(file)
    for line in file:
        line = line.strip()

        if not line:
            continue

        tokens = line.split(",", 1)

        try:
            rating = int(tokens[0].strip())
            review = tokens[1].strip().strip('"')
        except:
            continue

        ratings.append(rating)
        reviews.append(review)

def format_text(text):
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("'", "")

    return text

temp_reviews = []
for r in reviews:
    temp_text = format_text(r)
    temp_reviews.append(temp_text)

reviews = temp_reviews
y = np.array(ratings)
tfidf = TfidfVectorizer(max_features=2000, stop_words='english', max_df=0.7, min_df=5)
x = tfidf.fit_transform(reviews)

# Linear Regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train = x_train.toarray()
x_test = x_test.toarray()
x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

def rss(y, yhat):
    total = 0
    for i in range(len(y)):
        difference = y[i] - yhat[i]
        total += difference * difference

    return total

def BGD(x, y, learning_rate=0.01, num_epochs=500):
    num_samples = x.shape[0]
    num_features = x.shape[1]
    weights = np.zeros(num_features)

    for epoch in range(num_epochs):
        predictions = x @ weights
        residuals = y - predictions
        gradient = (-2/num_samples) * (x.T @ residuals)
        weights = weights - learning_rate * gradient

    return weights

weights = BGD(x_train, y_train)

def predict(x, weights):
    predictions = x @ weights
    predictions = np.round(predictions)
    predictions = np.clip(predictions, 1, 5)

    return predictions

y_train_linear_prediction = predict(x_train, weights)
y_test_linear_prediction = predict(x_test, weights)

def accuracy(y, yhat):
    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1

    return correct / len(y)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_train_logistic_prediction = model.predict(x_train)
y_test_logistic_prediction = model.predict(x_test)

cv = cross_val_score(model, x, y, cv=5)

# Results
print("\nLinear Regression:")
print("Training RSS:", rss(y_train, x_train @ weights))
print("Testing RSS:", rss(y_test, x_test @ weights))
print("Training Accuracy:", accuracy(y_train, y_train_linear_prediction))
print("Testing Accuracy:", accuracy(y_test, y_test_linear_prediction))

print("\nLogistic Regression:")
print("Training Accuracy:", accuracy(y_train, y_train_logistic_prediction))
print("Testing Accuracy:", accuracy(y_test, y_test_logistic_prediction))
print("Cross Validation Score:", cv)
print("Average Cross Validation Score:", np.mean(cv))