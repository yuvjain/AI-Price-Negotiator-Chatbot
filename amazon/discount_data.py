import pandas as pd
import pickle


# Load the saved model
with open('discount_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Load the saved DataFrame
df = pd.read_pickle('discount_data.pkl')


# Function to get discounted price based on inputs
def get_discounted_price(product_id, quantity, proposed_price):
    # Use the index as the product_id
    if product_id not in df.index:
        return "Product not found"
   
    # Retrieve the product based on the index
    product = df.loc[product_id]
   
    # Prepare the input features for the model
    input_features = product[['units_sold', 'rating', 'shipping_option_price', 'retail_price', 'cost_price']]
   
    # Convert the input features to a DataFrame to retain column names
    input_features_df = pd.DataFrame([input_features.values], columns=input_features.index)
   
    # Predict the discounted price using the trained model
    predicted_discounted_price = model.predict(input_features_df)[0]
   
    # Ensure the discounted price is above the cost price
    discounted_price = max(predicted_discounted_price, product['cost_price'])
   
    return discounted_price


# Example usage
product_id = 12  # Replace with actual product_id (index value)
quantity = 1  # Replace with actual quantity
proposed_price = 5.0  # Replace with actual proposed price


discounted_price = get_discounted_price(product_id, quantity, proposed_price)
print(f"Discounted Price for Product ID {product_id}: {discounted_price}")
