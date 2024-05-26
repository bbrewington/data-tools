from faker import Faker
import random
import pandas as pd

fake = Faker()
Faker.seed(8675309)

def generate_users(n):
    # Generate User Data
    users = []
    for _ in range(n):
        user = {
            'user_id': fake.uuid4(),
            'username': fake.user_name(),
            'email': fake.email()
        }
        users.append(user)
    return users

def generate_products(n):
    # Generate Product Data
    products = []
    for _ in range(n):
        product = {
            'product_id': fake.uuid4(),
            'name': fake.word(),
            'price': random.uniform(10, 100)
        }
        products.append(product)
    
    return products

def generate_orders(n, users, products):
    # Generate Order Data
    orders = []
    for _ in range(n):
        order = {
            'order_id': fake.uuid4(),
            'user_id': random.choice(users)['user_id'],
            'product_id': random.choice(products)['product_id'],
            'quantity': random.randint(1, 5)
        }
        orders.append(order)
    
    return orders

def write_to_snowflake(users_df, products_df, orders_df):
    # Write Data to Snowflake
    users_df.to_sql('users', engine, index=False, if_exists='append')
    products_df.to_sql('products', engine, index=False, if_exists='append')
    orders_df.to_sql('orders', engine, index=False, if_exists='append')
    
if __name__ == '__main__':
    users = generate_users(500)
    products = generate_products(20)
    orders = generate_orders(10000, users, products)

    users_df, products_df, orders_df = (pd.DataFrame(x) for x in [users, products, orders])

import pandas as pd
from sqlalchemy import create_engine

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='.env')

user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
account = os.getenv('SNOWFLAKE_ACCOUNT')
database = os.getenv('SNOWFLAKE_DATABASE')
schema = os.getenv('SNOWFLAKE_SCHEMA')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')

# Assume df is your DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})

# Create an SQLAlchemy engine
engine = create_engine(
    f'snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}'
)

# Use the to_sql method to write the DataFrame to a Snowflake table
df.to_sql(
    name='<your_table>',
    con=engine,
    index=False,
    if_exists='replace'  # Use 'append' if you want to append the data instead of replacing it
)
