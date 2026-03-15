# Databricks notebook source
df = spark.table("ecommerce_sales")

display(df)

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("bronze_ecommerce")

# COMMAND ----------

spark.table("bronze_ecommerce").display()

# COMMAND ----------

from pyspark.sql.functions import col

silver_df = spark.table("bronze_ecommerce")

silver_df = silver_df.dropna(subset=["CustomerID"])

silver_df = silver_df.withColumn(
    "TotalPrice",
    col("Quantity") * col("UnitPrice")
)

display(silver_df)

# COMMAND ----------

silver_df.write.format("delta").mode("overwrite").saveAsTable("silver_ecommerce")

# COMMAND ----------

spark.table("silver_ecommerce").display()

# COMMAND ----------

from pyspark.sql.functions import count, sum, avg

gold_df = spark.table("silver_ecommerce").groupBy("CustomerID").agg(
    count("InvoiceNo").alias("total_orders"),
    sum("TotalPrice").alias("total_spent"),
    avg("TotalPrice").alias("avg_order_value")
)

display(gold_df)

# COMMAND ----------

gold_df.write.format("delta").mode("overwrite").saveAsTable("gold_customer_features")

# COMMAND ----------

spark.table("gold_customer_features").display()

# COMMAND ----------

from pyspark.sql.functions import when

ml_df = spark.table("gold_customer_features")

ml_df = ml_df.withColumn(
    "high_value_customer",
    when(ml_df.total_spent > 5000, 1).otherwise(0)
)

display(ml_df)

# COMMAND ----------

pdf = ml_df.toPandas()

pdf.head()

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = pdf[["total_orders","total_spent","avg_order_value"]]
y = pdf["high_value_customer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# COMMAND ----------

import mlflow

with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)

    print("Accuracy:", accuracy)

# COMMAND ----------

