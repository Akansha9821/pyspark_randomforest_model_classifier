# Databricks notebook source
# MAGIC %md
# MAGIC ### Project Name: Tredence-CC_Integration
# MAGIC
# MAGIC ##### Project Information
# MAGIC - **Nestle SPOCs:** Sudarshan Ramesh
# MAGIC - **Created By:** Akansha Rana, Aman Chandel
# MAGIC - **Networks:** NPW, Frozen, Ambient, Chilled
# MAGIC
# MAGIC ##### Purpose
# MAGIC The purpose of this project is to build a classification model for the DRP Salvage project. The project involves the following steps:
# MAGIC - Data preparation
# MAGIC - Feature engineering
# MAGIC - Model training & evaluation
# MAGIC - Fail-safe implementation
# MAGIC - Model Testing
# MAGIC
# MAGIC ##### Acceptance Criteria
# MAGIC - The model should be able to identify the accuracy of classified Executed Recommendation.
# MAGIC - Building a random forest model and predict the probability of classification. If the classification recommended is '1' and the probability is >80%, then the value will be taken as 1. Otherwise, it will be taken as 0.
# MAGIC - For this model, we need columns like Time duration before move (days), Value of move. After that, train the model for daily DRP output and merge predictions with DRP output recommendations from 2023 to 2024 data and then save the record into Azure Data Lake Storage (ADLS).
# MAGIC
# MAGIC ##### Training Recommendation Input Paths
# MAGIC - **From 2023-2024:** `solutions/ift/ift/outbound/Tredence_CC_Integration/Outbound/training_data`
# MAGIC - **Daily Training Recommendation Input:** `solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound`
# MAGIC
# MAGIC ##### Output Folder
# MAGIC - `solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/DRP_All_Model_Predection/`
# MAGIC

# COMMAND ----------

# importing required modules & packages
import os
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import MulticlassMetrics

# COMMAND ----------

# MAGIC %run "/Workspace/Shared/NUSA/SupplyChain/Tredence-CC_Integration/AzureConnections"

# COMMAND ----------

# MAGIC %md
# MAGIC ##########################################################################
# MAGIC ###                  Input Training  Data                              ###
# MAGIC ##########################################################################

# COMMAND ----------

npw_training_data_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Outbound/training_data/training_data_npw/"
)

frozen_training_data_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Outbound/training_data/training_data_frozen/"
)

ambient_training_data_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Outbound/training_data/training_data_ambient"
)

chilled_trainig_data_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Outbound/training_data/training_data_chilled"

)


# COMMAND ----------

# MAGIC %md
# MAGIC ##########################################################################
# MAGIC ###                  Daily Reccomandation Training  Data               ###
# MAGIC ##########################################################################

# COMMAND ----------

npw_training_data_daily = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/todays_recommendations/npw/"
)

frozen_training_data_daily = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/todays_recommendations/frozen/"
)

ambient_training_data_daily = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/todays_recommendations/ambient/"
)

chilled_trainig_data_daily = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/todays_recommendations/chilled/"

)


# COMMAND ----------

# Reading Training Data for All for Network Ambient, Chilled, Frozen, Chilled 
try:
    ambient_training_data = spark.read.format("delta").load(ambient_training_data_path)
    npw_training_data = spark.read.format("delta").load(npw_training_data_path)
    frozen_training_data = spark.read.format("delta").load(frozen_training_data_path)
    chilled_training_data = spark.read.format("delta").load(chilled_trainig_data_path)
except Exception as e:
    print(f"Error in loading nps data: {str(e)}")
    raise SystemExit(f"Exiting due to the error: {str(e)}")

# COMMAND ----------

# Reading Npw Training Data
try:
    ambient_daily_reccomdation_df = spark.read.format("delta").load(ambient_training_data_daily)
    frozen_daily_reccomdation_df = spark.read.format("delta").load(frozen_training_data_daily)
    npw_daily_reccomdation_df = spark.read.format("delta").load(npw_training_data_daily)
    chilled_daily_reccomdation_df = spark.read.format("delta").load(chilled_trainig_data_daily)
    
except Exception as e:
    print(f"Error in loading nps data: {str(e)}")
    raise SystemExit(f"Exiting due to the error: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##########################################################################
# MAGIC #               Model Training  Start                                   #
# MAGIC ##########################################################################

# COMMAND ----------

def process_columns(df):
    try:
        df = (
            df.withColumn("Report_Run_Date", f.col("Report_Run_Date").cast("date"))
            .withColumn("DispatchDate", f.col("DispatchDate").cast("date"))
            .withColumn("IsExecuted", f.col("IsExecuted").cast("int"))
            .withColumn(
                "Time_duration_before_move",
                f.datediff(f.col("DispatchDate"), f.col("Report_Run_Date")),
            )
            .withColumn("Value_of_move", f.col("QtyMovedInPUM") * f.col("MovedValue"))
        )
        df = df.select("DaystoSalvage", "Distance", "Value_of_move", "IsExecuted")
        assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol='features')  
        assembler_df = assembler.transform(df).select('features', 'IsExecuted')
        return assembler_df
    except Exception as e:
        print("Error occurred while processing columns:", e)

# COMMAND ----------

def oversample_minority(train):
    try:
        minority_class = train.filter(f.col("IsExecuted") == 1)
        majority_class = train.filter(f.col("IsExecuted") == 0)
        oversampling_ratio = majority_class.count() / minority_class.count()
        oversample_minority_class = minority_class.sample(withReplacement=True, fraction=float(oversampling_ratio) / 3, seed=42)
        return majority_class.unionAll(oversample_minority_class)
    except Exception as e:
        print("Error occurred during oversampling:", e)

# COMMAND ----------

def train_model(train):
    try:
        rf = RandomForestClassifier(
            numTrees=70, maxDepth=10, featuresCol="features", labelCol="IsExecuted", seed=42
        )

        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction", labelCol="IsExecuted", metricName="areaUnderROC"
        )
        model = rf.fit(train)
        return model, evaluator
    except Exception as e:
        print("Error occurred during model training:", e)


# COMMAND ----------

def evaluate_model(model, test, evaluator):
    try:
        main_pred = model.transform(test)
        accuracy = evaluator.evaluate(main_pred)
        return accuracy, main_pred
    except Exception as e:
        print("Error occurred during model evaluation:", e)
        return None, None

# COMMAND ----------

def calculate_confusion_matrix(main_pred):
    try:
        preds_float= main_pred.select("IsExecuted", "prediction").rdd.map(lambda row: (row["prediction"], float(row["IsExecuted"])))
        cm = MulticlassMetrics(preds_float)
        confusion_matrix = pd.DataFrame(cm.confusionMatrix().toArray(),
                                         columns= ["actual negative", "actual positive"],
                                         index= ["predicted negative", "predicted positive"])
        return confusion_matrix
    except Exception as e:
        print("Error occurred while calculating confusion matrix:", e)


# COMMAND ----------

npw_processed_df = process_columns(npw_training_data)
frozen_processed_df = process_columns(frozen_training_data)
ambient_processed_df = process_columns(ambient_training_data)
chilled_procedded_df = process_columns(chilled_training_data)

# COMMAND ----------

train_npw, test_npw = npw_processed_df.randomSplit([0.50, 0.50], seed=42)
train_frozen, test_frozen = frozen_processed_df.randomSplit([0.50, 0.50], seed=42)
train_ambient, test_ambient = ambient_processed_df.randomSplit([0.50, 0.50], seed=42)
train_chilled, test_chilled = chilled_procedded_df.randomSplit([0.50, 0.50], seed=42)

print("NPW train count", train_npw.count())
print("NPW test count", test_npw.count())

print("Frozen train count", train_frozen.count())
print("Frozen test count", test_frozen.count())

print("Ambient train count", train_ambient.count())
print("Ambient test count", test_ambient.count())

print("Chilled train count", train_chilled.count())
print("Chilled test count", test_chilled.count())

# COMMAND ----------

balanced_train_npw = oversample_minority(train_npw)
balanced_train_frozen = oversample_minority(train_frozen)
balanced_train_ambient = oversample_minority(train_ambient)
balanced_train_chilled = oversample_minority(train_chilled)


print("NPW Balanced DF Count", balanced_train_npw.count())
print("Frozen Balanced DF Count", balanced_train_frozen.count())
print("Ambient Balanced DF Count", balanced_train_ambient.count())
print("Chilled Balanced DF Count", balanced_train_chilled.count())

# COMMAND ----------

model_npw, evaluator_npw = train_model(balanced_train_npw)
model_frozen, evaluator_frozen = train_model(balanced_train_frozen)
model_ambient, evaluator_ambient = train_model(balanced_train_ambient)
model_chilled, evaluator_chilled = train_model(balanced_train_chilled)

# COMMAND ----------

npw_accuracy, npw_main_pred = evaluate_model(model_npw, test_npw, evaluator_npw)
frozen_accuracy, frozen_main_pred = evaluate_model(model_frozen, test_frozen, evaluator_frozen)
ambient_accuracy, ambient_main_pred = evaluate_model(model_ambient, test_ambient, evaluator_ambient)
chilled_accuracy, chilled_main_pred = evaluate_model(model_chilled, test_chilled, evaluator_chilled)

print("Npw Accuracy", npw_accuracy)
print("Frozen Accuracy", frozen_accuracy)
print("Ambient Accuracy", ambient_accuracy)
print("Chilled Accuracy", chilled_accuracy)


# COMMAND ----------


confusion_matrix_npw = calculate_confusion_matrix(npw_main_pred)
confusion_matrix_frozen = calculate_confusion_matrix(frozen_main_pred)
confusion_matrix_ambient = calculate_confusion_matrix(ambient_main_pred)
confusion_matrix_chilled = calculate_confusion_matrix(chilled_main_pred)

print("NPW Confusion Matrix")
print(confusion_matrix_npw)

print("Frozen Confusion Matrix")
print(confusion_matrix_frozen)

print("Ambient Confusion Matrix")
print(confusion_matrix_ambient)

print("Chilled Confusion Matrix")
print(confusion_matrix_chilled)

# COMMAND ----------

# MAGIC %md
# MAGIC ##########################################################################
# MAGIC ###            Saving Model Predection               ###
# MAGIC ##########################################################################

# COMMAND ----------

npw_saving_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/DRP_All_Model_Predection/"
)

frozen_saving_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/DRP_All_Model_Predection/"
)

ambient_saving_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/DRP_All_Model_Predection/"
)

chilled_saving_path = (
    "abfss://root@" + stgAct_dev + ".dfs.core.windows.net/solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/DRP_All_Model_Predection/"

)


# COMMAND ----------

npw_input_daily=process_columns(npw_daily_reccomdation_df)
frozen_input_daily = process_columns(frozen_daily_reccomdation_df)
ambient_input_daily = process_columns(ambient_daily_reccomdation_df)
chilled_input_daily = process_columns(chilled_daily_reccomdation_df)

# COMMAND ----------

def daily_prediction(model,data_daily):
    try:
        daily_pred = model.transform(data_daily)
        return daily_pred
    except Exception as e:
        print("Error occurred during model evaluation:", e)
        return None

# COMMAND ----------

ouput_npw = daily_prediction(model_npw, npw_input_daily)
ouput_frozen = daily_prediction(model_frozen, frozen_input_daily)
output_ambient = daily_prediction(model_ambient, ambient_input_daily)
output_chilled = daily_prediction(model_chilled, chilled_input_daily)

# COMMAND ----------

ouput_npw=ouput_npw.dropDuplicates()
ouput_frozen=ouput_frozen.dropDuplicates()
output_ambient=output_ambient.dropDuplicates()
output_chilled=output_chilled.dropDuplicates()

# COMMAND ----------

def join_dfs(df, df2):
    try:
        df = (
            df.withColumn("Report_Run_Date", f.col("Report_Run_Date").cast("date"))
            .withColumn("DispatchDate", f.col("DispatchDate").cast("date"))
            .withColumn("IsExecuted", f.col("IsExecuted").cast("int"))
            .withColumn(
                "Time_duration_before_move",
                f.datediff(f.col("DispatchDate"), f.col("Report_Run_Date")),
            )
            .withColumn("Value_of_move", f.col("QtyMovedInPUM") * f.col("MovedValue"))
        )
        df = df.drop("IsExecuted")
        assembler = VectorAssembler(
            inputCols=["DaystoSalvage", "Distance", "Value_of_move"],
            outputCol="features",
        )
        assembler_df = assembler.transform(df)

        output = assembler_df.join(df2, on="features")
        output = output.select(
            [
                "MaterialID_batchoutput",
                "Batch",
                "SourceDC",
                "DestinationDC",
                "Report_Run_Date",
                "ArrivalDate",
                "DispatchDate",
                "ProductionDate",
                "ExpirationDate",
                "QtyMovedInPUM",
                "MovedValue",
                "DaystoSalvage",
                "WeekstoSalvage",
                "ShelfLifeRemainingPercent",
                "Freshness",
                "Distance",
                "MovedWeight",
                "ProductPlanningUnitsPerCase",
                "QtyMovedInCases",
                "SalvageDate",
                "IsExecuted",
                "prediction",
            ]
        )
        return output
    except Exception as e:
        print("Error occurred while processing columns:", e)

# COMMAND ----------

# npw_joined_df.count()

# COMMAND ----------

npw_joined_df = join_dfs(npw_daily_reccomdation_df, ouput_npw)
frozen_joined_df = join_dfs(frozen_daily_reccomdation_df, ouput_frozen)
ambient_joined_df = join_dfs(ambient_daily_reccomdation_df, output_ambient)
chilled_joined_df = join_dfs(chilled_daily_reccomdation_df, output_chilled)

# COMMAND ----------

npw_joined_df.filter(col("Prediction") == 1).count()
frozen_joined_df.filter(col("Prediction") == 1).count()
ambient_joined_df.filter(col("Prediction") == 1).count()
chilled_joined_df.filter(col("Prediction") == 1).count()

# COMMAND ----------

npw_joined_df.filter(col("IsExecuted") == 1).count()
frozen_joined_df.filter(col("IsExecuted") == 1).count()
ambient_joined_df.filter(col("IsExecuted") == 1).count()
chilled_joined_df.filter(col("IsExecuted") == 1).count()

# COMMAND ----------

def save_to_adls(df, file_name, file_path):
    try:
        df.write \
        .format("delta") \
        .mode("overwrite") \
        .save(os.path.join(file_path, file_name))
    except Exception as e:
        print(f"Error in saving data to ADLS: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")

# COMMAND ----------

npw_save_prediction = save_to_adls(npw_joined_df, file_name="npw_prediction", file_path=npw_saving_path)
frozen_save_prediction = save_to_adls(frozen_joined_df, file_name="frozen_prediction", file_path=frozen_saving_path)
ambient_save_prediction = save_to_adls(ambient_joined_df, file_name="ambient_prediction", file_path=ambient_saving_path)
chilled_save_prediction = save_to_adls(chilled_joined_df, file_name="chilled_prediction", file_path=chilled_saving_path)

# COMMAND ----------


