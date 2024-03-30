import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

movie_data = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

movie_data.dropna(inplace=True)
print(movie_data.head())
print(movie_data.info())
print(movie_data.shape)

plt.figure(figsize=(8, 6))
plt.hist(movie_data['Rating'], bins=20, color='orange')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
movie_data['Genre'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Distribution of Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

X = movie_data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = movie_data['Rating']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
