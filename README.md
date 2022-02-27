
# ML Pipeline on Databricks

Building a machine learning pipeline on Databricks using Spark for a bank marketing use case. The dataset has information about a phone call based marketing campaign that a bank carried out for persuading customers to subscribe a term deposit. The task is to analyse it and identify the patterns that will help finding conclusions in order to develop future strategies to get customers subscribed to a term deposit. Basically, this is going to be a binary classification problem where the goal is to predict if a client will subscribe or not (yes or no).



## Data

- Source --> University Of California Irvine Machine Learning Repository. [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

- Description --> bank-full.csv contains 17 attributes 45,211 records. Attributes contain 7 numerical features, 9 categorical features, and 1 dependent (target) feature. [attribute details](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

- DataLake --> Amazon S3 Bucket. [link](https://s3.console.aws.amazon.com/s3/buckets/bankmarket1611?region=us-east-1&tab=objects)
## Tools
- Databricks Comunity Edition
- Spark SQL, Spark MLlib
- Amazon S3 Bucket, AWS IAM
- pySpark:- OneHotEncoder, StringIndexer, VectorAssembler, Pipeline, LogisticRegression, BinaryClassificationEvaluator
## Overview

### Mounting S3 Bucket and Fetching Data
- Accessed Data from S3 Bucket using AWS IAM.
- Created dafaframe for python
- Created temporary table for sql

### Exploratory Analysis
- SQL Queries
- Databricks visualization
- Got gist of the data
- Drawn Insights

### Feature Engineering Pipeline
- Built Pipeline Stages
- Used StringIndexer on Categorical Features
- Used OneHotEncoder on Categorical Features
- Assembled Features to Vector

### Model Training
- Used Logistic Regression for Classification

### Prediction & Accuracy
- BinaryClassificationEvaluator
- Achieved AUC of 0.91 & Accuracy of 0.90
## Appendix
[Notebook]()
