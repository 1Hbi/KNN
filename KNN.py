# Import necessary libraries
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load dataset from a CSV file
dataset = pd.read_csv("car.data")

# Display the first few rows of the dataset
print(dataset.head())

# Initialize a LabelEncoder for encoding categorical variables
le = preprocessing.LabelEncoder()

# Encode categorical features using LabelEncoder
buying = le.fit_transform(dataset['buying'])
maint = le.fit_transform(dataset['maint'])
door = le.fit_transform(dataset['door'])
lug_boot = le.fit_transform(dataset['lug_boot'])
safety = le.fit_transform(dataset['safety'])
persons = le.fit_transform(dataset['persons'])
clss = le.fit_transform(dataset['class'])

# Specify the target variable
predict = "class"

# Create feature matrix (X) and target variable vector (y)
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clss)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Initialize and train the K-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)

# Evaluate the model's accuracy on the test set
acc = model.score(x_test, y_test)
print("Accuracy:", acc)

# Make predictions on the test set
predictions = model.predict(x_test)

# Define class labels for better interpretation of predictions
class_labels = ["unacc", "acc", "good", "vgoof"]

# Display predictions and actual values
for x in range(len(predictions)):
    print("Predicted:", class_labels[predictions[x]], "Data:", x_test[x], "Actual:", class_labels[y_test[x]])

    # Optionally print the k-neighbors for each prediction
    n = model.kneighbors([x_test[x]], n_neighbors=7)
    """print('N:', n)"""
