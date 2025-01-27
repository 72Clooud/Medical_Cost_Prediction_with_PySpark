from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from typing import List

class FeatureEngineer:
    
    def __init__(self, df: DataFrame):
        self.df = df
        
    def string_indexer(self, input_cols: List[str], output_cols: List[str]) ->  "FeatureEngineer":
        string_indexer = StringIndexer(inputCols=input_cols, outputCols=output_cols)
        string_indexer = string_indexer.fit(self.df)
        self.df = string_indexer.transform(self.df)
        return self
        
    def One_hot_encoder(self, input_col: List[str], output_col: List[str]) ->  "FeatureEngineer": 
        one_hot_encoder = OneHotEncoder(inputCols=input_col, outputCols=output_col)
        one_hot_encoder = one_hot_encoder.fit(self.df)
        self.df = one_hot_encoder.transform(self.df)
        return self
        
    def convert_to_binary(self, cols_to_binary: List[str]) -> "FeatureEngineer":
        for column in cols_to_binary:
            self.df = self.df.withColumn(column, col(column).cast('int'))
        return self
    
    def charges_to_flaot(self):
        self.df = self.df.withColumn('charges', col('charges').cast('double'))
        return self
    
    def assemble_numerical_features(self, cols: List[str], output_col_name: str) ->  "FeatureEngineer":
        assembler = VectorAssembler(inputCols=cols, outputCol=output_col_name)
        self.df = assembler.transform(self.df)
        return self
    
    def normalize_fetures(self, numerical_col_vector: str, output_name: str) -> "FeatureEngineer":
        scaler = StandardScaler(inputCol=numerical_col_vector, outputCol=output_name,
                                withStd=True, withMean=True)
        scaler = scaler.fit(self.df)
        self.df = scaler.transform(self.df)
        return self
    
        
    def get_DataFrame(self) -> DataFrame:
        return self.df