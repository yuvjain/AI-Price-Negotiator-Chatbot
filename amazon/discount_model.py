import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


# Load your dataset
df = pd.read_csv('summer-products-with-rating-and-performance_2020-08 .csv')


# Assuming 'price' column is the cost price in this dataset
df.rename(columns={'price': 'cost_price'}, inplace=True)


# Filter out rows where retail_price is less than cost_price
df = df[df['retail_price'] >= df['cost_price']]


# Define weights for the formula
MIN_PROF_MARGIN = 0.1  # Example value for minimum profit margin
X1 = 0.5  # Example value for X1
X2 = 0.8  # Example value for X2
X3 = 0.7  # Example value for X3


# Calculate Discount Percentage
VELOCITY_FACTOR = df['units_sold'] / df['units_sold'].max()  # Normalize units_sold
LIFECYLE_STAGE_FACTOR = df['rating'] / df['rating'].max()  # Normalize rating
STOCK_Q_FACTOR = df['shipping_option_price'] / df['shipping_option_price'].max()  # Normalize shipping_option_price
df['discount_percentage'] = MIN_PROF_MARGIN * (X1 * VELOCITY_FACTOR) * (X2 * LIFECYLE_STAGE_FACTOR) / (X3 * STOCK_Q_FACTOR)


# Ensure the discounted price is above the cost price
df['discounted_price'] = df['cost_price'] + (df['cost_price'] * MIN_PROF_MARGIN)


# Display the updated DataFrame
print("Training DataFrame:")
print(df[['retail_price', 'cost_price', 'discount_percentage', 'discounted_price']].head())


# Define X (features) and y (target variable)
X = df[['units_sold', 'rating', 'shipping_option_price', 'retail_price', 'cost_price']]  # Including cost_price as a feature
y = df['discounted_price']  # Target variable


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest Regressor model
model = RandomForestRegressor()


# Train the model
model.fit(X_train, y_train)


# Save the model to a file
with open('discount_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# Save the DataFrame to a file for later use
df.to_pickle('discount_data.pkl')


# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


r_squared = r2_score(y_test, y_pred)
print("R-squared Score:", r_squared)


# Visualizations
plt.figure(figsize=(12, 6))


# Heatmap of correlations
plt.subplot(1, 2, 1)
sns.heatmap(df[['units_sold', 'rating', 'shipping_option_price', 'retail_price', 'cost_price', 'discounted_price']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')


# Scatter plot of actual vs predicted discounted prices
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Discounted Price')
plt.ylabel('Predicted Discounted Price')
plt.title('Actual vs Predicted Discounted Prices')


plt.tight_layout()
plt.show()


# Print the accuracy of the model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
