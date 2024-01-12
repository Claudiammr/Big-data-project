from pyspark.sql.functions import when, col
from pyspark.sql import SparkSession as SpSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import argparse

def main(file_paths):
    # Create a Spark session
    spark = SpSession.builder \
        .appName("FlightsDelays") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Read and concatenate the CSV files
    df = None
    for file_path in file_paths:
        temp_df = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(file_path)
        df = temp_df if df is None else df.union(temp_df)

    # columns to eliminate
    columns = [
        "ArrTime",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "Diverted",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "TailNum",
        "TaxiOut",
        "Cancelled",
        "CancellationCode"
    ]

    # Eliminate columns
    df = df.drop(*columns)

    # We drop some columns that are higly correlated with another

    df = df.drop("CRSElapsedTime", "CRSElapsedTime", "CRSArrTime", "CRSDepTime")

    # Iterate over all columns in the DataFrame and replace "NA" with None
    for column in df.columns:
        df = df.withColumn(column, when(col(column) == "NA", None).otherwise(col(column)))

    # Drop rows with at least one missing value

    df = df.dropna()
    df = df.dropDuplicates()

    #Transform some variables

    # List of columns to exclude from conversion
    exclude_columns = ['UniqueCarrier', 'Origin', 'Dest']

    # Convert all columns to integer type except the ones in exclude_columns
    for column in df.columns:
        if column not in exclude_columns:
            df = df.withColumn(column, col(column).cast("integer"))

    # Read the CSV file
    usa_airport_df = spark.read.csv("us-airports.csv", header=True)

    # Select only the 'latitude_deg', 'longitude_deg', and 'iata_code' columns
    usa_airport_df = usa_airport_df.select("latitude_deg", "longitude_deg", "iata_code")

    # Join df with usa_airport_df
    # We join on df.Origin == usa_airport_df.iata_code
    # Using 'left' join to keep all records from df and add matching records from usa_airport_df
    # Join on Origin
    joined_df_origin = df.join(usa_airport_df, df.Origin == usa_airport_df.iata_code, 'inner') \
        .select(df["*"], usa_airport_df["latitude_deg"],
                usa_airport_df["longitude_deg"])

    joined_df_origin = (joined_df_origin.withColumnRenamed("latitude_deg","Origin_Lat")
                        .withColumnRenamed("longitude_deg", "Origin_Long"))




    # Join on Dest
    final_df = joined_df_origin.join(usa_airport_df, joined_df_origin.Dest == usa_airport_df.iata_code, 'inner')

    final_df = (final_df.withColumnRenamed("latitude_deg", "Dest_Lat")
                .withColumnRenamed("longitude_deg", "Dest_Long")).drop("iata_code")

    df = final_df.dropna().dropDuplicates()

    # Create a StringIndexer
    indexer = StringIndexer(inputCol="UniqueCarrier", outputCol="UniqueCarrierIndex")

    # Fit the indexer to the DataFrame and transform it
    df_indexed = indexer.fit(df).transform(df)

    # Create a OneHotEncoder
    encoder = OneHotEncoder(inputCols=["UniqueCarrierIndex"], outputCols=["UniqueCarrierVec"])

    # Apply the encoder to the DataFrame
    df_encoded = encoder.fit(df_indexed).transform(df_indexed)

    final_df = df_encoded.drop("UniqueCarrier")

    # Convert "Origini_Lat" from string to double
    final_df = final_df.withColumn("Origin_Lat", final_df["Origin_Lat"].cast("double"))

    # Convert "Origini_Long" from string to double
    final_df = final_df.withColumn("Origin_Long", final_df["Origin_Long"].cast("double"))

    # Convert "Dest_Lat" from string to double
    final_df = final_df.withColumn("Dest_Lat", final_df["Dest_Lat"].cast("double"))

    # Convert "Dest_Long" from string to double
    final_df = final_df.withColumn("Dest_Long", final_df["Dest_Long"].cast("double"))

    final_df.show()
    final_df.printSchema()

    '''
    from pyspark.sql import SparkSession
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.feature import VectorAssembler

    # Assuming your target variable is "ArrDelay" and your feature columns are selected
    feature_columns = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "FlightNum", "DepDelay", "Distance", "Origin_Lat",
                       "Origin_Long", "Dest_Lat", "Dest_Long", "UniqueCarrierIndex"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(final_df)

    train_data, test_data = assembled_data.randomSplit([0.7, 0.3], seed=123)

    lr = LinearRegression(featuresCol="features", labelCol="ArrDelay", regParam=0.01)
    lr_model = lr.fit(train_data)

    test_results = lr_model.evaluate(test_data)

    # Print evaluation metrics
    print("Root Mean Squared Error (RMSE):", test_results.rootMeanSquaredError)
    print("R-squared (R2):", test_results.r2)
    '''
    # For classification we create a new Variable ArrDelayClass
    # 1. Define classes based on ArrDelay
    # Categorize flights as "Not Late" (0) if ArrDelay <= 0, and "Late" (1) otherwise
    final_df = final_df.withColumn("ArrDelayClass", when(final_df["ArrDelay"] <= 0, 0).otherwise(1))

    # As is imbalanced we do an oversampling
    df_a = final_df.filter(final_df['ArrDelayClass'] == 1)
    df_b = final_df.filter(final_df['ArrDelayClass'] == 0)

    a_count = df_a.count()
    b_count = df_b.count() 
    ratio = a_count / b_count

    df_b_oversampled = df_b.sample(withReplacement=True, fraction=ratio, seed=1)
    combined_df = df_a.unionAll(df_b_oversampled)

    # Preparation of the data for the model
    columns = [
    "ArrDelay"
    ]
    
    # Eliminate columns
    new_combined_df = combined_df.drop(*columns)
    
    feature_columns = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "FlightNum", "DepDelay", "UniqueCarrierIndex"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(new_combined_df)
    train_data, test_data = assembled_data.randomSplit([0.7, 0.3], seed=123)

    # Linear Regression Model

    lr = LogisticRegression(labelCol="ArrDelayClass", featuresCol="features",regParam=0.01)
    
    # Fit the model
    lrModel = lr.fit(train_data)
    
    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    predictions = lrModel.transform(test_data)
    
    # AUC-ROC
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="ArrDelayClass")
    auc = evaluator.evaluate(predictions)
    print(f"AUC-ROC: {auc:.4f}")

    multi_evaluator = MulticlassClassificationEvaluator(labelCol="ArrDelayClass", predictionCol="prediction")
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Random Forest
    # set rf model 
    rf = RandomForestClassifier(labelCol="ArrDelayClass", featuresCol="features")
    
    # instantiate pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # train model
    model_rf = rf.fit(train_data)

    # create prediction column on test data
    results = model_rf.transform(test_data)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="ArrDelayClass", metricName="accuracy")
    
    # Evaluate the model
    accuracy = evaluator.evaluate(results)
    print("Test set accuracy = {:.2f}".format(accuracy))


    # Cross validation
    # Define the hyperparameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()

    # Create the cross-validator
    cross_validator = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                    evaluator=MulticlassClassificationEvaluator(labelCol="ArrDelayClass", metricName="accuracy"),
                              numFolds=5, seed=123)
    column=["features"]
    train_data=train_data.drop(*column)
    cv_model = cross_validator.fit(train_data)

    best_rf_model = cv_model.bestModel.stages[-1]
    importances = best_rf_model.featureImportances
    
    
    print("Feature Importances:")
    for feature, importance in zip(feature_columns, importances):
        print(f"{feature}: {importance:.4f}")
    
        # Make predictions on the test data
    predictions = cv_model.transform(test_data)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="ArrDelayClass", metricName="accuracy")
    
    # Evaluate the model
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = {:.2f}".format(accuracy))
        
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='PySpark Flights Delays')
    parser.add_argument("file_paths", nargs='*', type=str, help='Paths to the CSV files')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with the file paths
    main(args.file_paths)


