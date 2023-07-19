from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

######################
# Reading a CSV file #
######################

prices = spark.read.options(header="true").csv("mnt/data_lake/landing/prices.csv")

prices.show()

from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    FloatType,
    IntegerType,
    DateType,
)

######################
# Enforcing a Schema #
######################

schema = StructType(
    [
        StructField("store", StringType(), nullable=False),
        StructField("countrycode", StringType(), nullable=False),
        StructField("brand", StringType(), nullable=False),
        StructField("price", FloatType(), nullable=False),
        StructField("currency", StringType(), nullable=True),
        StructField("quantity", IntegerType(), nullable=True),
        StructField("date", DateType(), nullable=False),
    ]
)
prices = (
    spark.read.options(header="true")
    .schema(schema)
    .csv("mnt/data_lake/landing/prices.csv")
)
print(prices.dtypes)
# [
#     ("store", "string"),
#     ("countrycode", "string"),
#     ("brand", "string"),
#     ("price", "float"),
#     ("currency", "string"),
#     ("quantity", "int"),
#     ("date", "date"),
# ]

#################
# Cleaning Data #
#################

prices = spark.read.options(header="true").csv("landing/prices.csv")
prices.show()

# Handling invalid rows

# mode (default PERMISSIVE): allows a mode for dealing with corrupt records during parsing.
#   PERMISSIVE : sets other fields to null when it meets a corrupted record,
#     and puts the malformed string into a new field configured by columnNameOfCorruptRecord.
#     When a schema is set by user, it sets null for extra fields.
#   DROPMALFORMED : ignores the whole corrupted records.
#   FAILFAST : throws an exception when it meets corrupted records.
prices = spark.read.options(header="true", mode="DROPMALFORMED").csv(
    "landing/prices.csv"
)

# Supplying default values for missing data

prices.fillna(25, subset=["quantity"]).show()

# Conditionally replace values

from pyspark.sql.functions import col, when
from datetime import date, timedelta
from pyspark.sql.functions import col, avg, count

one_year_from_now = date.today().replace(year=date.today().year + 1)
better_frame = employees.withColumn(
    "end_date",
    when(col("end_date") > one_year_from_now, None).otherwise(col("end_date")),
)
better_frame.show()

#####################
# Transforming Data #
#####################

# Filtering and ordering rows
prices_in_belgium = prices.filter(col("countrycode") == "BE").orderBy(col("date"))

# Selecting and renaming columns (and remove duplicates)
prices.select(col("store"), col("brand").alias("brandname")).distinct()

# Grouping and aggregating with mean()
(prices.groupBy(col("brand")).mean("price")).show()

# Grouping and aggregating with agg()
(
    prices.groupBy(col("brand")).agg(
        avg("price").alias("average_price"),
        count("brand").alias("number_of_items")
    )
).show()

# Joining related data
ratings_with_prices = ratings.join(prices, ["brand", "model"])

############################
# Packaging an application #
############################

import os
os.system("""
          spark-submit \
            --master 'local[*]' \
            --py-files PY_FILES \
            MAIN_PYTHON_FILE \
            app_arguments
          """)

# Collecting all dependencies in one archive

# pydiaper
#     cleaning
#         clean_prices.py
#         clean_ratings.py
#         __init__.py
#     data_catalog
#         catalog.py

os.system("""
  zip \
    --recurse-paths \
      dependencies.zip \
      pydiaper
      """)

os.system("""
          spark-submit \
            --py-files dependencies.zip \
              pydiaper/cleaning/clean_prices.py
          """)
