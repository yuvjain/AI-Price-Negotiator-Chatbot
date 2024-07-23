# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv('/home/yuv/miniproject/amazon/summer-products-with-rating-and-performance_2020-08.csv')

# Assuming 'price' column is the cost price in this dataset
df.rename(columns={'price': 'cost_price'}, inplace=True)

# Filter out rows where retail_price is less than cost_price
df = df[df['retail_price'] >= df['cost_price']]

# Define weights for the formula
MIN_PROF_MARGIN = 0.5  # Example value for minimum profit margin
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

#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

ddf=pd.read_csv('/home/yuv/miniproject/amazon/test_dataset.csv')

def get_discounted_price(product_id, quantity, proposed_price):
    product = ddf[ddf['product_id'] == product_id]
    if product.empty:
        return "Product not found"
    
    # Prepare the input features for the model
    input_features = product[['units_sold', 'rating', 'shipping_option_price', 'retail_price', 'cost_price']].values[0].reshape(1, -1)
    
    # Predict the discounted price using the trained model
    predicted_discounted_price = model.predict(input_features)[0]
    
    # Ensure the discounted price is above the cost price
    discounted_price = max(predicted_discounted_price, product['cost_price'].values[0])
    
    return predicted_discounted_price

# Function to negotiate the discounted price
def negotiate_discounted_price(product_id, quantity, proposed_price):
    discounted_price = get_discounted_price(product_id, quantity, proposed_price)
    
    # Define negotiation threshold (e.g., ±5% of the discounted price)
    threshold = 0.05
    lower_bound = discounted_price * (1 + threshold)
    pp = float(proposed_price)
    
    # Negotiate the price
    if pp >= lower_bound:
        return proposed_price  # Accept the proposed price if it's above the lower bound
    else:
        return lower_bound  # Counter with the lower bound price

class ActionGetDisPrice(Action):

    def name(self) -> Text:
        return "action_get_dis_price"

    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        
        product_id = tracker.get_slot("product_id")
        quantity = tracker.get_slot("quantity")
        proposed_price = tracker.get_slot("proposed_price")
        pp = float(proposed_price)
        
        
        discount_price=get_discounted_price(product_id,quantity,proposed_price)
        discount_price = discount_price*(X1+X2)/X3
        final_price=negotiate_discounted_price(product_id,quantity,proposed_price)
        
        if discount_price < float(proposed_price): 
            return [SlotSet("discount_price",proposed_price)]
        else:
            return [SlotSet("discount_price",discount_price)]

        # ddf = pd.read_csv("C:\\Users\\Lenovo\\Documents\\rasa_projects\\datasets\\client_inventory.csv")

        
        # product = ddf.loc[product_id]
    
        # # Prepare the input features for the model
        # input_features = product[['units_sold', 'rating', 'shipping_option_price', 'retail_price', 'cost_price']]
    
        # # Convert the input features to a DataFrame to retain column names
        # input_features_df = pd.DataFrame([input_features.values], columns=input_features.index)
    
        # # Predict the discounted price using the trained model
        # predicted_discounted_price = model.predict(input_features_df)[0]
    
        # # Ensure the discounted price is above the cost price
        # discounted_price = max(predicted_discounted_price, product['cost_price'])

        # return [SlotSet("discount_price","discounted_price")]

        
        #NOTES:
        #- add action to domain. add slot to domain
        #- change stories to add action call
        #- create secondary loop till user says yes and in turn,
        #- create accept_offer and deny_offer entities
        #- dont forget to activate the rasa action server 