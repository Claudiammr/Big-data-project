from pyspark.sql.functions import when, col, format_string 
from pyspark.sql import SparkSession as SpSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

import argparse


# Function to convert 'hhmm' formatted string to minutes
def convert_to_minutes(column_name):
    return ((col(column_name).substr(1, 2).cast("int") * 60) +
            col(column_name).substr(3, 2).cast("int"))


def main(file_paths):
    # Create a Spark session
    spark = SpSession.builder \
        .appName("FlightsDelays") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # spark.sparkContext.setLogLevel("WARN")

    # Read and concatenate the CSV files
    df = None
    for file_path in file_paths:
        temp_df = spark.read.format("csv").option("inferSchema", "true").option("delimiter", ",").option("header",
                                                                                                         "true").load(
            file_path)
        df = temp_df if df is None else df.union(temp_df)

    # Sampling
    df = df.sample(fraction=0.0005, seed=42)


    print(f"Running on {df.count()} instances...")

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
        "LateAircraftDelay"
    ]

    # Eliminate columns
    df = df.drop(*columns)

    # Drop canceled flights (where "Cancelled" column is not equal to 1)
    df = df.filter(df["Cancelled"] != 1)

    columns = [
        "TailNum",
        "TaxiOut",
        "Cancelled",
        "CancellationCode"
    ]

    df = df.drop(*columns)

    # We drop some columns that are higly correlated with another

    # df = df.drop("CRSElapsedTime", "CRSElapsedTime", "CRSArrTime", "CRSDepTime")
    
    df.printSchema()

    # Iterate over all columns in the DataFrame and replace "NA" with None
    for column in df.columns:
        df = df.withColumn(column, when(col(column) == "NA", None).otherwise(col(column)))

    # Drop rows with at least one missing value

    df = df.dropna()
    df = df.dropDuplicates()

    # Transform some variables

    # Convert 'Distance', 'DepDelay','CRSElapsedTime', and 'ArrDelay' from string to integer
    df = df.withColumn("Distance", col("Distance").cast("integer"))
    df = df.withColumn("DepDelay", col("DepDelay").cast("integer"))
    df = df.withColumn("ArrDelay", col("ArrDelay").cast("integer"))
    df = df.withColumn("CRSElapsedTime", col("CRSElapsedTime").cast("integer"))

    # Convert 'CRSDepTime' and 'CRSArrTime' from integer to string
    df = df.withColumn("CRSDepTime", col("CRSDepTime").cast("string"))
    df = df.withColumn("CRSArrTime", col("CRSArrTime").cast("string"))

    # Apply the conversion to each time column
    df = df.withColumn("CRSDepTime", convert_to_minutes("CRSDepTime"))
    df = df.withColumn("CRSArrTime", convert_to_minutes("CRSArrTime"))
    df = df.withColumn("DepTime", convert_to_minutes("DepTime"))


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

    joined_df_origin = (joined_df_origin.withColumnRenamed("latitude_deg", "Origin_Lat")
                        .withColumnRenamed("longitude_deg", "Origin_Long"))

    # Join on Dest
    final_df = joined_df_origin.join(usa_airport_df, joined_df_origin.Dest == usa_airport_df.iata_code, 'inner')

    final_df = (final_df.withColumnRenamed("latitude_deg", "Dest_Lat")
                .withColumnRenamed("longitude_deg", "Dest_Long")).drop("iata_code")

    df = final_df.dropna().dropDuplicates()

    # Convert "Origini_Lat" from string to double
    df = df.withColumn("Origin_Lat", df["Origin_Lat"].cast("int"))

    # Convert "Origini_Long" from string to double
    df = df.withColumn("Origin_Long", df["Origin_Long"].cast("int"))

    # Convert "Dest_Lat" from string to double
    df = df.withColumn("Dest_Lat", df["Dest_Lat"].cast("int"))

    # Convert "Dest_Long" from string to double
    df = df.withColumn("Dest_Long", df["Dest_Long"].cast("int"))

    # Create a StringIndexer
    indexer = StringIndexer(inputCol="UniqueCarrier", outputCol="UniqueCarrierIndex")

    # Create a OneHotEncoder
    encoder = OneHotEncoder(inputCols=["UniqueCarrierIndex"], outputCols=["UniqueCarrierVec"])

    assembler = VectorAssembler(
        inputCols=["Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime",
                   "FlightNum", "CRSElapsedTime", "ArrDelay", "DepDelay", "Distance",
                   "FlightNum", "CRSElapsedTime", "DepDelay", "Distance",
                   "Origin_Lat", "Origin_Long", "Dest_Lat", "Dest_Long", "UniqueCarrierVec"],
        outputCol="features")

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    # Define a pipeline
    pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])

    # Fit the pipeline to the DataFrame and transform it
    df = pipeline.fit(df).transform(df)

    df.show()
    df.printSchema()

    # For classification we create a new Variable ArrDelayClass based on ArrDelay
    # Categorize flights as "Not Late" (0) if ArrDelay <= 0, and "Late" (1) otherwise
    df = df.withColumn("ArrDelayClass", when(df["ArrDelay"] <= 0, 0).otherwise(1))
    print(df.count())

    #### MODELING

    # Split the data
    (train_data, test_data) = df.randomSplit([0.8, 0.2])
    print("train_data", train_data.count())
    # As is imbalanced we do an oversampling
    df_a = train_data.filter(col("ArrDelayClass") == 1)
    a = df_a.count()
    print(a)
    df_b = train_data.filter(col("ArrDelayClass") == 0)
    b = df_b.count()
    print(b)


    ratio = a / b
    print("ratio", ratio)
    df_b_oversampled = df_b.sample(withReplacement=True, fraction=ratio, seed=1)
    combined_df = df_a.unionAll(df_b_oversampled)
    
    class_counts = combined_df.groupBy("ArrDelayClass").count().orderBy("ArrDelayClass")
    class_counts.show()

    # Preparation of the data for the model
    columns = [
        "ArrDelay"
    ]

    # Eliminate columns
    balanced_train_data = combined_df.drop(*columns)

    # Models
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="ArrDelayClass", regParam=0.01)
    # rf = RandomForestClassifier(featuresCol="features", labelCol="ArrDelayClass")
    # dt = DecisionTreeRegressor(featuresCol="features", labelCol="ArrDelay")
    # gbt = GBTRegressor(featuresCol="features", labelCol="ArrDelay", maxIter=10)

    # Train Models
    lr_model = lr.fit(balanced_train_data)
    # rf_model = rf.fit(balanced_train_data)
    # dt_model = dt.fit(train_data)
    # gbt_model = gbt.fit(train_data)

    # Predictions
    lr_predictions = lr_model.transform(test_data)
    # rf_predictions = rf_model.transform(test_data)
    # dt_predictions = dt_model.transform(test_data)
    # gbt_predictions = gbt_model.transform(test_data)

    # Evaluation for classification models
    binary_evaluator = BinaryClassificationEvaluator(labelCol="ArrDelayClass")

    # Multiclass Classification Evaluator
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="ArrDelayClass")

    # Evaluate Logistic Regression
    lr_auc = binary_evaluator.evaluate(lr_predictions)
    lr_precision = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "precisionByLabel"})
    lr_recall = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "recallByLabel"})
    lr_f1 = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "f1"})
    lr_area_under_pr = binary_evaluator.setMetricName("areaUnderPR").evaluate(lr_predictions)

    print("Logistic Regression AUC:", lr_auc)
    print("Logistic Regression Precision:", lr_precision)
    print("Logistic Regression Recall:", lr_recall)
    print("Logistic Regression F1 Score:", lr_f1)
    print("Logistic Regression Area Under PR:", lr_area_under_pr)


    # Evaluate Random Forest Classifier
    rf_auc = binary_evaluator.evaluate(rf_predictions)
    rf_precision = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "precisionByLabel"})
    rf_recall = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "recallByLabel"})
    rf_f1 = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "f1"})
    rf_area_under_pr = binary_evaluator.setMetricName("areaUnderPR").evaluate(rf_predictions)

    print("Random Forest Classifier AUC:", rf_auc)
    print("Random Forest Classifier Precision:", rf_precision)
    print("Random Forest Classifier Recall:", rf_recall)
    print("Random Forest Classifier F1 Score:", rf_f1)
    print("Random Forest Classifier Area Under PR:", rf_area_under_pr)

    from pyspark.ml.evaluation import RegressionEvaluator

    # Regression Evaluator
    regression_evaluator = RegressionEvaluator(labelCol="ArrDelay")

    # Evaluate Decision Tree Regressor
    dt_rmse = regression_evaluator.evaluate(dt_predictions)
    dt_mae = regression_evaluator.setMetricName("mae").evaluate(dt_predictions)
    dt_r2 = regression_evaluator.setMetricName("r2").evaluate(dt_predictions)

    print("Decision Tree Regressor RMSE:", dt_rmse)
    print("Decision Tree Regressor MAE:", dt_mae)
    print("Decision Tree Regressor R-squared:", dt_r2)

    # Evaluate GBT Regressor
    gbt_rmse = regression_evaluator.evaluate(gbt_predictions)
    gbt_mae = regression_evaluator.setMetricName("mae").evaluate(gbt_predictions)
    gbt_r2 = regression_evaluator.setMetricName("r2").evaluate(gbt_predictions)

    print("GBT Regressor RMSE:", gbt_rmse)
    print("GBT Regressor MAE:", gbt_mae)
    print("GBT Regressor R-squared:", gbt_r2)

    # Cross validation
    print("Final DF Shape: ")
    df.printSchema()

    # For classification we create a new Variable ArrDelayClass based on ArrDelay
    # Categorize flights as "Not Late" (0) if ArrDelay <= 15, and "Late" (1) otherwise
    df = df.withColumn("ArrDelayClass", when(df["ArrDelay"] <= 15, 0).otherwise(1))

    #### MODELING

    # Split the data
    (train_data, test_data) = df.randomSplit([0.8, 0.2])

    dt = DecisionTreeRegressor(featuresCol="features", labelCol="ArrDelay")

    # Models
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="ArrDelayClass", regParam=0.01)
    rf = RandomForestClassifier(featuresCol="features", labelCol="ArrDelayClass")
    
    gbt = GBTRegressor(featuresCol="features", labelCol="ArrDelay", maxIter=10)

    print("Training...")
    # Train Models
    lr_model = lr.fit(train_data)
    rf_model = rf.fit(train_data)
    dt_model = dt.fit(train_data)
    gbt_model = gbt.fit(train_data)

    print("Done.")

    print("Evaluating...")
    # Predictions
    lr_predictions = lr_model.transform(test_data)
    rf_predictions = rf_model.transform(test_data)
    dt_predictions = dt_model.transform(test_data)
    gbt_predictions = gbt_model.transform(test_data)

    print("Done.")


    # Evaluation for classification models
    binary_evaluator = BinaryClassificationEvaluator(labelCol="ArrDelayClass")

    # Multiclass Classification Evaluator
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="ArrDelayClass")


    # Evaluate Logistic Regression
    lr_auc = binary_evaluator.evaluate(lr_predictions)
    lr_precision = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "precisionByLabel"})
    lr_recall = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "recallByLabel"})
    lr_f1 = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "f1"})
    lr_area_under_pr = binary_evaluator.setMetricName("areaUnderPR").evaluate(lr_predictions)

    print("Logistic Regression AUC:", lr_auc)
    print("Logistic Regression Precision:", lr_precision)
    print("Logistic Regression Recall:", lr_recall)
    print("Logistic Regression F1 Score:", lr_f1)
    print("Logistic Regression Area Under PR:", lr_area_under_pr)

    
    # Evaluate Random Forest Classifier
    rf_auc = binary_evaluator.evaluate(rf_predictions)
    rf_precision = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "precisionByLabel"})
    rf_recall = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "recallByLabel"})
    rf_f1 = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "f1"})
    rf_area_under_pr = binary_evaluator.setMetricName("areaUnderPR").evaluate(rf_predictions)

    print("Random Forest Classifier AUC:", rf_auc)
    print("Random Forest Classifier Precision:", rf_precision)
    print("Random Forest Classifier Recall:", rf_recall)
    print("Random Forest Classifier F1 Score:", rf_f1)
    print("Random Forest Classifier Area Under PR:", rf_area_under_pr)

    


    # Regression Evaluator
    regression_evaluator = RegressionEvaluator(labelCol="ArrDelay")

    # Evaluate Decision Tree Regressor
    dt_rmse = regression_evaluator.evaluate(dt_predictions)
    dt_mae = regression_evaluator.setMetricName("mae").evaluate(dt_predictions)
    dt_r2 = regression_evaluator.setMetricName("r2").evaluate(dt_predictions)

    print("Decision Tree Regressor RMSE:", dt_rmse)
    print("Decision Tree Regressor MAE:", dt_mae)
    print("Decision Tree Regressor R-squared:", dt_r2)



    # Evaluate GBT Regressor
    gbt_rmse = regression_evaluator.evaluate(gbt_predictions)
    gbt_mae = regression_evaluator.setMetricName("mae").evaluate(gbt_predictions)
    gbt_r2 = regression_evaluator.setMetricName("r2").evaluate(gbt_predictions)

    print("GBT Regressor RMSE:", gbt_rmse)
    print("GBT Regressor MAE:", gbt_mae)
    print("GBT Regressor R-squared:", gbt_r2)


        # Cross validation
        # Define the hyperparameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [10, 20, 30]) \
            .addGrid(rf.maxDepth, [5, 10, 15]) \
            .build()

        # Create the cross-validator
        cross_validator = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                        
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
