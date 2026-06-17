import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/wine.csv')

X = df[['alcohol']] 
y = df['quality']

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='actual')
plt.plot(X_test, y_pred, color='red', label='predicted')
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.title('Linear Regression: alcohol vs quality')
plt.legend()
plt.savefig('regression_result.png')
plt.show()

print("係数:", model.coef_)
print("切片:", model.intercept_)