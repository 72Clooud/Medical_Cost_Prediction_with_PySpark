from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

class DataLoader:
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.spark = SparkSession.builder.master('local[*]').appName('Mediacal_cost_prediction').getOrCreate()
        self.spark_consext = self.spark.sparkContext

    def load_data(self) -> DataFrame:
        df = self.spark.read.csv(self.file_path, header=True, inferSchema=True)
        return df

    def remove_outliers(self, df: DataFrame, outliers_dict: dict) -> DataFrame:
        conditions = [
            (col(col_name) >= min_val) & (col(col_name) <= max_val)
            for col_name, (min_val, max_val) in outliers_dict.items()
        ]
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition &= condition
        df = df.filter(combined_condition)
        return df