##################################
# Writing Unit Tests for PySpark #
##################################

import pyspark as spark
from pyspark.sql import Row
from pyspark.sql.functions import col

purchase = Row("price", "quantity", "product")
record = purchase(12.99, 1, "cake")
record_multiple = (
  purchase(12.99, 1, "cake"),
  purchase(2.99, 2, "cookies")
)
df = spark.createDataFrame((record,))
df2 = spark.createDataFrame(record_multiple)

def link_with_exchange_rates(prices, rates):
    return prices.join(rates, ["currency", "date"])
  
def calculate_unit_price_in_euro(df):
    return df.withColumn(
      "unit_price_in_euro",
      col("price") / col("quantity") * col("exchange_rate_to_euro"))

unit_prices_with_ratings = (
    calculate_unit_price_in_euro(
        link_with_exchange_rates(prices, exchange_rates)
    )
)
