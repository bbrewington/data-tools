-- CREATE WAREHOUSE cube_demo_wh;
-- CREATE DATABASE cube_demo;
CREATE SCHEMA brentbrewington.cube_demo_ecom;

CREATE TABLE brentbrewington.cube_demo_ecom.line_items
( id INTEGER,
  order_id INTEGER,
  product_id INTEGER,
  price INTEGER,
  created_at TIMESTAMP
);

COPY INTO brentbrewington.cube_demo_ecom.line_items (id, order_id, product_id, price, created_at)
FROM 's3://cube-tutorial/line_items.csv'
FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1);

CREATE TABLE brentbrewington.cube_demo_ecom.orders
( id INTEGER,
  user_id INTEGER,
  status VARCHAR,
  completed_at TIMESTAMP,
  created_at TIMESTAMP
);

COPY INTO brentbrewington.cube_demo_ecom.orders (id, user_id, status, completed_at, created_at)
FROM 's3://cube-tutorial/orders.csv'
FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1);

CREATE TABLE brentbrewington.cube_demo_ecom.users
( id INTEGER,
  user_id INTEGER,
  city VARCHAR,
  age INTEGER,
  gender VARCHAR,
  state VARCHAR,
  first_name VARCHAR,
  last_name VARCHAR,
  created_at TIMESTAMP
);

COPY INTO brentbrewington.cube_demo_ecom.users (id, city, age, gender, state, first_name, last_name, created_at)
FROM 's3://cube-tutorial/users.csv'
FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1);

CREATE TABLE brentbrewington.cube_demo_ecom.products
( id INTEGER,
  name VARCHAR,
  product_category VARCHAR,
  created_at TIMESTAMP
);

COPY INTO brentbrewington.cube_demo_ecom.products (id, name, created_at, product_category)
FROM 's3://cube-tutorial/products.csv'
FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1);
