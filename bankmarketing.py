# Databricks notebook source
# MAGIC %md
# MAGIC ### Mounting S3 Bucket and Fetching Data

# COMMAND ----------

dbutils.fs.unmount("/mnt/s3data")

# COMMAND ----------

import urllib
ACCESS_KEY = "AKIAUGUB6AB6DDNXCR57" #inactivated for security
SECRET_KEY = "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" #inactivated for security
ENCODED_SECRET_KEY = urllib.parse.quote(SECRET_KEY, "")
AWS_BUCKET_NAME = "bankmarket1611"
MOUNT_NAME = "s3data"
dbutils.fs.mount("s3n://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
display(dbutils.fs.ls("/mnt/s3data"))

# COMMAND ----------

file_location= "dbfs:/mnt/s3data/bank-full.csv"
file_type= "csv"

infer_schema= "true"
first_row_is_header= "true"
delimiter= ";"

df = spark.read.load(file_location,
                     format=file_type, 
                     sep=delimiter, 
                     inferSchema=True, 
                     header=first_row_is_header)

# COMMAND ----------

temp_table_name= "data"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Analysis

# COMMAND ----------

print(df.columns)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.stat.freqItems(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','day', 'month',  'poutcome', 'y'],0.5).collect()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT y, count(y)
# MAGIC FROM data
# MAGIC GROUP BY y

# COMMAND ----------

# MAGIC %md
# MAGIC > Data is unbalanced. Need AUC(Area Under Curve) instead of Accuracy to evaluate model.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT marital,y, count(marital)
# MAGIC FROM data
# MAGIC GROUP BY marital,y

# COMMAND ----------

# MAGIC %md
# MAGIC > Churning rate almost similar across marital category.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT education,y, count(education)
# MAGIC FROM data
# MAGIC GROUP BY education,y

# COMMAND ----------

# MAGIC %md
# MAGIC > Customer with secondary education is targeted the most. Rate of success is similar across education slab.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT y,housing, count(housing)
# MAGIC FROM data
# MAGIC GROUP BY y, housing

# COMMAND ----------

# MAGIC %md
# MAGIC > Customers with housing loan refused subscription more than customer without housing loan.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT y,contact, count(contact)
# MAGIC FROM data
# MAGIC GROUP BY y, contact

# COMMAND ----------

# MAGIC %md
# MAGIC > Percentage of refusal when contacted via cellular is less than percentage of subscription.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT y,poutcome, count(poutcome)
# MAGIC FROM data
# MAGIC GROUP BY y, poutcome

# COMMAND ----------

# MAGIC %md
# MAGIC >Customer who churned in previous marketing campaigns signed in for new subscription at a significantly higher number.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT y, avg(balance)
# MAGIC FROM data
# MAGIC GROUP BY y

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT y, avg(pdays)
# MAGIC FROM data
# MAGIC GROUP BY y

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering Pipeline

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
stages = []

# COMMAND ----------

#separating numerical features
numeric_features = [x[0] for x in df.dtypes if x[1] == 'int']
df.select(numeric_features).show(5)

# COMMAND ----------

categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

#encoding categorical features
for categorical_feature in categorical_features:
    stringIndexer = StringIndexer(inputCol = categorical_feature, outputCol = categorical_feature + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categorical_feature+"categorical"])
    
    #staging indexer and encoder for each categorical feature
    stages += [stringIndexer, encoder]

# COMMAND ----------

#label indexing dependent feature to 0 and 1
label_indexer = StringIndexer(inputCol = 'y', outputCol = 'label')
stages += [label_indexer]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler_input = [c+"categorical" for c in categorical_features] + numeric_features
assembler = VectorAssembler(inputCols=assembler_input, outputCol='features')
stages += [assembler]


# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages = stages)

pipelineModel = pipeline.fit(df)
df= pipelineModel.transform(df)

# COMMAND ----------

df_final= df.select('label', 'features')
display(df_final)

# COMMAND ----------

train, test = df_final.randomSplit([0.8, 0.2], seed = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)

model = log_reg.fit(train)

# COMMAND ----------

import matplotlib.pyplot as plt
train_summary = model.summary
roc = train_summary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('Training set ROC Curve')
plt.show()

# COMMAND ----------

train_accuracy = train_summary.accuracy
train_areaUnderROC= train_summary.areaUnderROC
print(" Training Accuracy: " , train_accuracy)
print(" Training Area under ROC: " , train_areaUnderROC)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction & Accuracy

# COMMAND ----------

predictions = model.transform(test)
predictions.select('label', 'prediction').show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator


evaluator = BinaryClassificationEvaluator()
test_areaUnderROC= evaluator.evaluate(predictions)
test_accuracy= model.evaluate(test).accuracy
print('Test accuracy', test_accuracy)
print('Test Area Under ROC', test_areaUnderROC)

# COMMAND ----------

# MAGIC %md
# MAGIC > AUC of 0.91 achieved ( AUC> 0.85 is a high classification accuracy ).
