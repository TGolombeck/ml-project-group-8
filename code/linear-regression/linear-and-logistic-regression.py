import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

reviews = []
ratings = []
with open("../../data/all_data.csv", "r", encoding='utf-8') as file:
    next(file)
    for line in file:
        line = line.strip()

        if not line:
            continue

        tokens = line.rsplit(",", 1)

        try:
            review = tokens[0].strip().strip('"')
            rating = int(float(tokens[1].strip()))
        except:
            continue

        reviews.append(review)
        ratings.append(rating)

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
x_train_linear = x_train.toarray()
x_test_linear = x_test.toarray()

mean = np.mean(x_train_linear, axis=0)
std = np.std(x_train_linear, axis=0)
std[std == 0] = 1

x_train_linear = (x_train_linear - mean) / std
x_test_linear = (x_test_linear - mean) / std
x_train_linear = np.hstack([np.ones((x_train_linear.shape[0], 1)), x_train_linear])
x_test_linear = np.hstack([np.ones((x_test_linear.shape[0], 1)), x_test_linear])

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

weights = BGD(x_train_linear, y_train)

def predict(x, weights):
    predictions = x @ weights
    predictions = np.round(predictions)
    predictions = np.clip(predictions, 1, 5)

    return predictions

y_train_linear_prediction = predict(x_train_linear, weights)
y_test_linear_prediction = predict(x_test_linear, weights)

def accuracy(y, yhat):
    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1

    return correct / len(y)

# Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)

y_train_logistic_prediction = model.predict(x_train)
y_test_logistic_prediction = model.predict(x_test)

cv = cross_val_score(model, x, y, cv=5)
cm = confusion_matrix(y_test, y_test_logistic_prediction)

# Results
print("\nLinear Regression:")
print("Training RSS:", rss(y_train, x_train_linear @ weights))
print("Testing RSS:", rss(y_test, x_test_linear @ weights))
print("Training Accuracy:", accuracy(y_train, y_train_linear_prediction))
print("Testing Accuracy:", accuracy(y_test, y_test_linear_prediction))

print("\nLogistic Regression:")
print("Training Accuracy:", accuracy(y_train, y_train_logistic_prediction))
print("Testing Accuracy:", accuracy(y_test, y_test_logistic_prediction))
print("Cross Validation Score:", cv)
print("Average Cross Validation Score:", np.mean(cv))
print("\nConfusion Matrix:")
print("Rows: Actual, Columns: Predicted")
print(cm)

# Linear Regression Table
"""
print("Actual \tPredicted")
for i in range(len(y_test)):
    print(f"\t{y_test[i]} \t{y_test_linear_prediction[i]}")
"""