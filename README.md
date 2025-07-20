# employeeSalary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Experience': [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
    'Education': ['Bachelor', 'Bachelor', 'Master', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'PhD', 'PhD'],
    'Job Title': ['Analyst', 'Analyst', 'Engineer', 'Engineer', 'Manager', 'Analyst', 'Engineer', 'Manager', 'Manager', 'Manager'],
    'Salary': [35000, 42000, 60000, 72000, 95000, 39000, 63000, 85000, 91000, 105000]
}

df = pd.DataFrame(data)
df.head()
# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Education', 'Job Title'], drop_first=True)

# Separate features and target variable
X = df_encoded.drop('Salary', axis=1)
y = df_encoded['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
new_data = pd.DataFrame({
    'Experience': [4],
    'Education_Master': [1],
    'Education_PhD': [0],
    'Job Title_Engineer': [1],
    'Job Title_Manager': [0]
})

predicted_salary = model.predict(new_data)
print("Predicted Salary: ₹", round(predicted_salary[0], 2))
plt.scatter(df['Experience'], df['Salary'], color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.grid(True)
plt.show()
