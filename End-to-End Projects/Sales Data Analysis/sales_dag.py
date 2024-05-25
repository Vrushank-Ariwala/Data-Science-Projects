from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Libraries for read_data()
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# Libraries for preprocess_data()
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType
# ssign full text values to priority levels libraries
from pyspark.sql.functions import when

# Libraries for feature_engineering()
from pyspark.sql import functions as F

# Libraries for bq_check_and_load_data()
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq

import json

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 0
}

# Define the DAG
dag = DAG(
    'sales_data_pipeline',
    default_args=default_args,
    description='A simple data pipeline for sales data analysis'
)


def read_data():

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName('sales_data_analysis') \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    data_file_path = r'/home/vrushankariwala/airflow_project/Data/data.csv'

    df = spark.read.option('header', 'true').csv(
        data_file_path, inferSchema=True)

    rd_json_strings = df.toJSON().collect()

    # Show the DataFrame
    return rd_json_strings


# Assign datatypes modules

# ssign full text values to priority levels modules


def preprocess_data():

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("JSON to DataFrame") \
        .getOrCreate()

    json_data = read_data()

    # Create RDD from the list of JSON strings
    pp_json_rdd = spark.sparkContext.parallelize(json_data)

    # Read JSON RDD into DataFrame
    df_pp = spark.read.json(pp_json_rdd)

    # Drop duplicate rows
    df_pp = df_pp.dropDuplicates()

    # Drop rows with missing values
    df_pp = df_pp.dropna()

    # Renaming the Columns
    columns = df_pp.columns
    new_columns = ['country', 'item_type', 'order_date', 'order_id', 'order_priority', 'region', 'sales_channel', 'ship_date',
                   'total_cost', 'total_profit', 'total_revenue', 'unit_cost', 'unit_price', 'units_sold']
    # new_columns = ['region', 'country', 'item_type', 'sales_channel', 'order_priority', 'order_date', 'order_id',
    #           'ship_date', 'units_sold', 'unit_price', 'unit_cost', 'total_revenue', 'total_cost', 'total_profit']
    for old_col, new_col in zip(columns, new_columns):
        df_pp = df_pp.withColumnRenamed(old_col, new_col)

    # Assigning appropriate datatypes
    func = udf(lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
    df_pp = df_pp.withColumn('order_date', func(col('order_date')))
    df_pp = df_pp.withColumn('ship_date', func(col('ship_date')))

    # Convert order_id column's datatype to string
    df_pp = df_pp.withColumn('order_id', col('order_id').cast('string'))

    # Assign full text values to priority levels
    df_pp = df_pp.withColumn(
        'order_priority',
        when(df_pp['order_priority'] == "H", 'High')
        .when(df_pp['order_priority'] == "M", 'Medium')
        .when(df_pp['order_priority'] == "L", 'Low')
        .when(df_pp['order_priority'] == "C", 'Critical')
        .otherwise(0)  # Default value if none of the above conditions are met
    )

    pp_json_strings = df_pp.toJSON().collect()

    # # Stop Spark session to release resources
    # spark.stop()

    # Show the DataFrame
    return pp_json_strings


def feature_engineering():

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("JSON to DataFrame") \
        .getOrCreate()

    strings = preprocess_data()

    # Create RDD from the list of JSON strings
    fe_json_rdd = spark.sparkContext.parallelize(strings)

    # Read JSON RDD into DataFrame
    df_fe = spark.read.json(fe_json_rdd)

    # Creating new shipping_time column
    df_fe = df_fe.withColumn(
        "shipping_time", F.datediff("ship_date", "order_date"))

    # Extract date components
    df_fe = df_fe.withColumn("order_month", F.month("order_date")) \
        .withColumn("order_year", F.year("order_date")) \
        .withColumn("order_weekday", F.dayofweek("order_date")) \
        .withColumn("ship_month", F.month("ship_date")) \
        .withColumn("ship_year", F.year("ship_date")) \
        .withColumn("ship_weekday", F.dayofweek("ship_date"))

    fe_json_strings = df_fe.toJSON().collect()

    return fe_json_strings


def bq_check_and_load_data():
    """
    This function checks if a BigQuery table is empty and loads data from a list of JSON strings.

    Args:
        client: A Google BigQuery client object.
        project_id: The ID of the GCP project containing the BigQuery table.
        table_id: The full path to the BigQuery table (project.dataset.table).
        c_strings: A list of JSON strings representing the data to be loaded.

    Returns:
        None
    """
    # Create a BigQuery client
    credentials = service_account.Credentials.from_service_account_file(
        r'/home/vrushankariwala/airflow_project/Data/keys.json')
    client = bigquery.Client(credentials=credentials,
                             project='sales-data-analysis-421208')

    # Define the project id and table id
    project_id = "sales-data-analysis-421208"
    table_id = 'sales-data-analysis-421208.Sales_Data_1_million_records.sales_table'

    bq_strings_data = feature_engineering()
    # Load JSON strings into a pandas DataFrame
    pandas_df = pd.DataFrame([json.loads(s) for s in bq_strings_data])

    # Check if the table is empty
    query = f"""
  SELECT COUNT(*) FROM `{table_id}`
  """
    query_job = client.query(query)
    results = query_job.result()  # Wait for the query to complete
    row = next(results)  # Get the first row (should be the count)
    if row[0] == 0:
        print("Table is empty, proceeding to load data.")
    else:
        print(f"Truncating table {table_id} before loading new data.")
        query = f"""
    TRUNCATE TABLE `{table_id}`
    """
        client.query(query)  # Execute the truncation query

    pandas_gbq.context.credentials = credentials

    # Load data to BigQuery
    try:
        pandas_gbq.to_gbq(pandas_df, table_id, project_id=project_id)
        print("Data loaded successfully to BigQuery!")
    except Exception as e:
        print(f"Error loading data: {e}")


# Define the tasks using PythonOperator
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=read_data,
    dag=dag,
)

preprocessing_task = PythonOperator(
    task_id='preprocessing_task',
    python_callable=preprocess_data,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=feature_engineering,
    dag=dag,
)

load_bq_task = PythonOperator(
    task_id='load_bq_task',
    python_callable=bq_check_and_load_data,
    dag=dag,
)


# Set task dependencies
load_data_task >> preprocessing_task >> feature_engineering_task >> load_bq_task
