# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create a larger sample dataset (50 records)
np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 50)
noise = np.random.normal(0, 5, 50)
exam_score = 5 * hours_studied + 30 + noise  # Linear relationship + noise

# Create DataFrame
df = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Exam_Score': exam_score
})

# Split data into input (X) and output (y)
X = df[['Hours_Studied']]
y = df['Exam_Score']

# Split into training and test sets (20% test -> 10 samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plot Actual vs Predicted (Prediction Accuracy Plot)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Grade Prediction Accuracy')
plt.legend()
plt.grid(True)
plt.show()
