import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

reviews = []
ratings = []
with open("./reviews.csv", "r") as file:
    for line in file:
        tokens = line.strip().rsplit(",", 1)
        review = tokens[0].strip('"')
        rating = int(tokens[1])

        reviews.append(review)
        ratings.append(rating)

def format_text(text):
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace(".", "")

    return text

temp_reviews = []
for r in reviews:
    temp_text = format_text(r)
    temp_reviews.append(temp_text)

reviews = temp_reviews
y = np.array(ratings)
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(reviews).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

def rss(y, yhat):
    total = 0
    for i in range(len(y)):
        difference = y[i] - yhat[i]
        total += difference * difference

    return total

def BGD(x, y, learning_rate=0.01, num_epochs=1000):
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

y_train_prediction = predict(x_train, weights)
y_test_prediction = predict(x_test, weights)

def accuracy(y, yhat):
    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1

    return correct / len(y)

print("Results:")
print("Training RSS:", rss(y_train, x_train @ weights))
print("Testing RSS:", rss(y_test, x_test @ weights))
print("Training Accuracy:", accuracy(y_train, y_train_prediction))
print("Testing Accuracy:", accuracy(y_test, y_test_prediction))