
#IRIS TASK
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
X = iris.data  
y = iris.target 
data = pd.read_csv('IRIS.csv')
print(data.sample(2))
print(data.columns)
print("Shape:", data.shape)
numeric_data = data.select_dtypes(include=np.number)
print("Mean:")
print(numeric_data.mean())
numeric_data = data.select_dtypes(include=np.number)
print("Median:")
print(numeric_data.median())
print("\nSum:")
print(data.sum())
print("\nMinimum:")
print(data.min())
print("\nMaximum:")
print(data.max())
print(data.iloc[5])
filtered_data = data.loc[data["species"] == "Iris-setosa"]
print(filtered_data)
plt.figure(figsize=(10, 6))
data['species'].value_counts().plot(kind='bar', color='green')
plt.title('Distribution of Iris Species')
plt.xlabel('Species')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
