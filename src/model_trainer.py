from pyspark.mllib.evaluation import RegressionMetrics

class ModelHandler:
    
    def __init__(self, train, test):
        self.train = train
        self.test = test        
        
    def train_model(self, regressor):
        reg = regressor.fit(self.train)
        return reg

    def convert_to_rdd(self, model):
        def transform_and_convert(df):
            pred_df = model.transform(df).select('prediction', 'charges').dropna()
            return pred_df.rdd.map(tuple)
        
        pred_train_rdd = transform_and_convert(self.train)
        pred_test_rdd = transform_and_convert(self.test)
        return pred_train_rdd, pred_test_rdd

    
    def evaluate_model(self, model_name, rdd_train_data, rdd_test_data) -> None:
        
        metrics_train = RegressionMetrics(rdd_train_data)
        metrics_test = RegressionMetrics(rdd_test_data)
        
        print(f"\nModel name: {model_name}")
        print("Training Data Metrics:")
        print(f"  MSE: {metrics_train.meanSquaredError}")
        print(f"  RMSE: {metrics_train.rootMeanSquaredError}")
        print(f"  MAE: {metrics_train.meanAbsoluteError}")
        print(f"  R2: {metrics_train.r2}")

        print("\nTesting Data Metrics:")
        print(f"  MSE: {metrics_test.meanSquaredError}")
        print(f"  RMSE: {metrics_test.rootMeanSquaredError}")
        print(f"  MAE: {metrics_test.meanAbsoluteError}")
        print(f"  R2: {metrics_test.r2}")