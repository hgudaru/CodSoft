import pandas as pd  # Importing the pandas library for data manipulation
import os  # Importing the os module for handling file paths
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import numpy as np  # Importing numpy for numerical operations
from sklearn.metrics import confusion_matrix  # Importing confusion_matrix from scikit-learn
import seaborn as sns  # Importing seaborn for enhanced data visualization
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting the dataset
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler for feature scaling
from sklearn.neighbors import KNeighborsClassifier  # Importing KNeighborsClassifier for KNN classification
from sklearn.metrics import accuracy_score, classification_report  # Importing evaluation metrics
from sklearn.datasets import load_iris  # Importing the Iris dataset for demonstration

# Writing Function to read a CSV file
def read_csv_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    try:
        # Loading CSV file
        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Writing Function to train a KNN classifier
def train_knn_classifier(X_train, y_train, k=3):
    # Standardize the features using StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    return knn_classifier, scaler

# Writing Function to predict the class of a flower based on attributes
def predict_flower_class(attributes, classifier, scaler):
    # Standardize the input attributes using the same scaler
    attributes_scaled = scaler.transform([attributes])
    
    # Making a prediction
    predicted_class = classifier.predict(attributes_scaled)
    return predicted_class[0]

# Replacing 'iris..csv' with the filename of your CSV file
csv_file_name = 'iris..csv'
csv_file_path = os.path.join(os.path.dirname(__file__), csv_file_name)
data_frame = read_csv_file(csv_file_path)

# Extracting features (sepal and petal length and width) and target variable (class)
X = data_frame[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
y = data_frame['class']

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the KNN classifier
knn_classifier, scaler = train_knn_classifier(X_train, y_train)

# Making predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Displaying classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred, zero_division=1))

# Example: Predicting the class of a flower with specific attributes 
sl = float(input("Enter Sepal Length: "))  #Taking Inputs
sw = float(input("Enter Sepal Width: "))
pl = float(input("Enter Petal Length: "))
pw = float(input("Enter Petal Width: "))
flower_attributes = [sl, sw, pl, pw]  # Replacing with your flower attributes
predicted_class = predict_flower_class(flower_attributes, knn_classifier, scaler)
print(f"Predicted Class for Flower: {predicted_class}")

# Writing Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Extracting features (sepal and petal length and width) and target variable (class)
X = data_frame[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
y = data_frame['class']

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the KNN classifier
knn_classifier, scaler = train_knn_classifier(X_train, y_train)

# Making predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Ploting the confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=np.unique(y))

# Loading Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['flower_name'] = iris.target_names[iris.target]

# Line plot for differentiating Sepal Length vs Sepal Width
sns.lineplot(x='sepal length (cm)', y='sepal width (cm)', hue='flower_name', data=iris_df, marker='o')
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Line plot for differentiating Petal Length vs Petal Width
sns.lineplot(x='petal length (cm)', y='petal width (cm)', hue='flower_name', data=iris_df, marker='o')
plt.title('Petal Length vs Petal Width')
plt.show()
