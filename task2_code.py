import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # Importing SMOTE for handling class imbalance

# Assuming the CSV file is in the same directory as the script
csv_file_name = 'creditcard.csv'
csv_file_path = os.path.join(os.path.dirname(__file__), csv_file_name)

# Loading the dataset
data = pd.read_csv(csv_file_path)

# Displaying basic overview of the dataset
print("Dataset Overview:")
print(data.head())
print(data.info())

# Separating legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
print("\nClass Distribution:")
print("Legitimate Transactions:", legit.shape)
print("Fraudulent Transactions:", fraud.shape)

# Using SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(data.drop('Class', axis=1), data['Class'])

# Creating a new dataset with resampled data
new_dataset = pd.concat([pd.DataFrame(X_resampled, columns=data.drop('Class', axis=1).columns), pd.Series(Y_resampled, name='Class')], axis=1)

# Separating features and target variable in the new dataset
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print("\nResampled Dataset Overview:")
print(X.head())
print("\nClass Distribution in Resampled Dataset:")
print(Y.value_counts())

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("\nData Splitting:")
print("Original Dataset Shape:", X.shape)
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Training a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating model performance on the training set
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('\nModel Training Performance:')
print('Accuracy on Training data:', training_data_accuracy)

# Evaluating model performance on the testing set
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('\nModel Testing Performance:')
print('Accuracy on Test data:', test_data_accuracy)

# Additional Analysis: Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(Y_test, X_test_prediction)
class_report = classification_report(Y_test, X_test_prediction)

print("\nAdditional Analysis:")
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)
