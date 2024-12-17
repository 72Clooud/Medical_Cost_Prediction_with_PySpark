from dotenv import load_dotenv
load_dotenv()
import os
os.environ['SPARK_HOME'] = os.getenv('SPARK_HOME')
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
os.environ["PYSPARK_DRIVER_PYTHON_OPTS"] = 'python'
os.environ['PYSPARK_PYTHON'] = "python"

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql.functions import col

# Init spark session
spark = SparkSession.builder.master('local[*]').appName('Mediacal_cost_prediction').getOrCreate()

df = spark.read.csv('./data/medical.csv', header=True, inferSchema=True)
df = df.filter(df['charges'] < 35000)
df = df.filter(df['bmi'] < 45)

# Take only categorical_cols
categorical_cols = [col for col in df.columns if col not in {'age', 'bmi', 'children', 'charges'}]

# Index categorial columns
string_indexer = StringIndexer(inputCols=categorical_cols,
                               outputCols=['sex_index', 'smoker_index', 'region_index'])
string_indexer = string_indexer.fit(df)
df = string_indexer.transform(df)

# Convert float values in above columns to int values
df = df.withColumn('sex_index', col('sex_index').cast('int'))
df = df.withColumn('smoker_index', col('smoker_index').cast('int'))

# One hot encoding on region column
encoder = OneHotEncoder(inputCol='region_index',
                        outputCol='region_one_hot')
encoder = encoder.fit(df)
df = encoder.transform(df)

# Create feature vector for training model
assembler = VectorAssembler(inputCols=['age', 'bmi', 'children'], 
                            outputCol='numerical_cols_vector')
df = assembler.transform(df)

# Normalize numerical values in feature vector column
scaler = StandardScaler(inputCol='numerical_cols_vector',
                        outputCol='scaled_numerical_cols_vector',
                        withStd=True, withMean=True)
scaler = scaler.fit(df)
df = scaler.transform(df)

# Concat all together
assembler = VectorAssembler(inputCols=['sex_index', 'smoker_index', 'region_one_hot', 'scaled_numerical_cols_vector'],
                            outputCol='final_features_vector')
df = assembler.transform(df)

# Spliting data into train and test 
train, test = df.randomSplit([0.7, 0.3])

# Train 3 diffrent Regressors to check the best performance
reg_BGT = GBTRegressor(featuresCol='final_features_vector', labelCol='charges')
reg_LR = LinearRegression(featuresCol='final_features_vector', labelCol='charges')
reg_RF = RandomForestRegressor(featuresCol='final_features_vector', labelCol='charges')

model_GBT = reg_BGT.fit(train)
model_RF = reg_RF.fit(train)
model_LR = reg_LR.fit(train)

# function for preparing data to right format 
def return_info_for_metrics(model, train, test):
    
    pred_train_df = model.transform(train)
    pred_test_df = model.transform(test)
    
    pred_and_actuals = pred_train_df[['prediction', 'charges']]
    pred_and_actuals_rdd = pred_and_actuals.rdd
    
    pred_and_actuals_rdd = pred_and_actuals_rdd.map(tuple)
    
    return pred_and_actuals_rdd

models = [model_GBT, model_RF, model_LR]
models_metric_names = ['metrics_GBT', 'metrics_RF', 'metrics_LR']

# create regression metrics for each model
for model, metrics_name in zip(models, models_metric_names):
    pred_and_actuals_rdd = return_info_for_metrics(model, train, test)
    metrics_name = RegressionMetrics(pred_and_actuals_rdd)
    print(f'mse: {metrics_name.meanSquaredError}\n \
          rmse: {metrics_name.rootMeanSquaredError}\n \
          mae: {metrics_name.meanAbsoluteError}\n \
          r2: {metrics_name.r2}')
    
# End session
spark.stop()
