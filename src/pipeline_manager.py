from typing import List
from pyspark.sql.functions import col
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer


class PipelineManager:
    
    def __init__(self, data_path):
        self.data_path = data_path
        
        
    def process_data(self, outliers_dict: dict,
                     string_indexer_cols: List[str],
                     string_indexer_output: str, 
                     cols_to_binary: List[str], 
                     cols_for_one_hot: List[str],
                     output_for_one_hot: List[str],
                     assemble_numerical_cols: List[str],
                     assmble_numerical_output_name: str,
                     cols_to_normalize: List[str],
                     normalized_output: str, 
                     final_cols: List[str],
                     final_output: str):
        
        data_loader = DataLoader(self.data_path)
        df = data_loader.load_data()
        df = data_loader.remove_outliers(df, outliers_dict)
            
        features_engineer = FeatureEngineer(df)        
        features_engineer.string_indexer(input_cols=string_indexer_cols, output_cols=string_indexer_output)
        features_engineer.convert_to_binary(cols_to_binary=cols_to_binary)
        features_engineer.charges_to_flaot()
        features_engineer.One_hot_encoder(input_col=cols_for_one_hot, output_col=output_for_one_hot)
        features_engineer.assemble_numerical_features(cols=assemble_numerical_cols, output_col_name=assmble_numerical_output_name)
        features_engineer.normalize_fetures(numerical_col_vector=cols_to_normalize, output_name=normalized_output)
        features_engineer.assemble_numerical_features(cols=final_cols, output_col_name=final_output)
        df = features_engineer.get_DataFrame()
        return df             