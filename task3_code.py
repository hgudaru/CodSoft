import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming the CSV file is in the same directory as the script
csv_file_name = 'Titanic-Dataset.csv'
csv_file_path = os.path.join(os.path.dirname(__file__), csv_file_name)

# Loading the dataset
data = pd.read_csv(csv_file_path)

# Explore the dataset
print(data.head())

# Drop unnecessary columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical features to numerical
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Check column names and data types
print(data.dtypes)

# Plotting graphs for survival based on different features
plt.figure(figsize=(18, 12))

# Plot 1: Survival based on Passenger Class
plt.subplot(2, 2, 1)
sns.countplot(x='Pclass', hue='Survived', data=data, palette='viridis', edgecolor='black', linewidth=2)

# Plot 2: Survival based on Sex
plt.subplot(2, 2, 2)
sns.countplot(x='Sex_male', hue='Survived', data=data, palette='Set2', edgecolor='black', linewidth=2)
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])

# Plot 3: Survival based on Embarked Location
plt.subplot(2, 2, 3)
sns.countplot(x='Embarked_S', hue='Survived', data=data, palette='coolwarm', edgecolor='black', linewidth=2)
plt.xticks(ticks=[0, 1], labels=['C', 'S'])

# Plot 4: Survival based on Sibling/Spouse count
plt.subplot(2, 2, 4)
sns.countplot(x='SibSp', hue='Survived', data=data, palette='muted', edgecolor='black', linewidth=2)

plt.show()

# Function to predict survival
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex_male': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked_Q': [0],  # Assuming 'Embarked_Q' is not provided in the input
        'Embarked_S': [1] if embarked == 'S' else [0],
    })

    # Ensure the order of features matches the order during training
    input_data = input_data[X.columns]

    # Make the prediction
    prediction = clf.predict(input_data)
    return prediction[0]

# Example usage
pclass = int(input("Enter passenger class (1, 2, 3): "))
sex = int(input("Enter sex (0 for female, 1 for male): "))
age = float(input("Enter age: "))
sibsp = int(input("Enter number of siblings/spouses aboard: "))
parch = int(input("Enter number of parents/children aboard: "))
fare = float(input("Enter fare: "))
embarked = input("Enter embarked location (C, Q, S): ")

prediction = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
if prediction == 1:
    print("The passenger is predicted to survive.")
else:
    print("The passenger is predicted not to survive.")
