# Random Forest Classifier in PySpark - Lab

## Introduction  

In this lab, we would like to a Random Forest Classifier model to study over the ecommerce behavior from a multi-category store. First, we need to download the data to your local machine, then we will load the data from the local machine onto a Pandas Dataframe.

## Objectives  

* Use the kaggle eCommerce dataset in the previous lab and reimplement using PySpark

## Instruction
* Accept the kaggle policy and download the data from here https://www.kaggle.com/code/tshephisho/ecommerce-behaviour-using-xgboost/data
* For the first model building, we'll only use the 2019-Nov csv data (which is still around ~2gb zipped)


```python
# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```


```python
import matplotlib.pyplot as plt
import squarify
import matplotlib.dates as dates
from datetime import datetime

%matplotlib inline
```


```python
from pyspark.sql import SparkSession #entry point for pyspark

#instantiate spark instance
spark = SparkSession.builder.appName('Random Forest eCommerce').master("local[*]").getOrCreate()
```


```python
path = "" #wherever path you saved the kaggle file to
df = spark.read.csv(path, header=True, inferSchema=True)
df.printSchema() #to see the schema
```

We've already used this dataset, but feel free to explore around. Now, we want to use the pandas instead of pyspark, we have to use the `action` functions, which then means there will be a network shuffle. Earlier lab used the Iris dataset which was about ~1KB, but the current dataset may be too large, and may throw an `OutOfMemory` error if we attempt to load the data into the pandas dataframe. I would suggest only take few rows for exploratory analysis if pandas is more comfortable library. Otherwise, sticking with native PySpark functions would be much better option. 


```python
# pd.DataFrame(df.take(10), columns=df.columns).transpose()
```

### Know your Customers

How many unique customers visit the site?


```python
# using naitve pyspark..
from pyspark.sql.functions import countDistinct
df.select(countDistinct("user_id")).show() 
```

Did you notice the spark progress bar when you triggered the `action` function? The `show()` function is the `action` function which means the lazy evaluation of Spark was triggered and completed a certain job. `read.csv` should have been another job. If you go to the `localhost:4040` you should be able to see 2 completed jobs under the `Jobs` tab, which are `csv` and `showString`. While a heavy job is getting executed, you can take a look at the `Executors` tab to examine the executors completing the tasks in parellel. Now, we may not see if we run this on a local machine, but this behavior should definitely be visible if you're on a cloud system, such as EMR.

### (Optional) Visitors Daily Trend

Does traffic flunctuate by date? Try using the event_time and user_id to see traffic, and draw out the plots for visualization.


```python
# try cleaning out the event_time column then using groupby/count
# import pyspark.sql.functions as F
# use this as a reference to clean the event_time column
# https://stackoverflow.com/questions/67827631/how-udf-function-works-in-pyspark-with-dates-as-arguments 
```

Question: We would still like to see the cart abandonment rate using the dataset. What relevant features can we use for modeling?


```python
# your answer
```

Now, let's build out the model.


```python
from pyspark.ml.feature import VectorAssembler

#columns you'd like to use
feature_cols = []
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
df.show()
```

Is there a labeler column that we'd like to use?


```python
from pyspark.ml.feature import StringIndexer

#what should we use for the inputCol here?
labeler = StringIndexer(inputCol='', outputCol='encoded')
df = labeler.fit(df).transform(df)
df.show()
```

Now build the train/test dataset.


```python
train, test = df.randomSplit([0.7, 0.3], seed=42)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
```


```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='', labelCol='encoded')
model = rf.fit(train)
predictions = model.transform(test)
# what goes in the select() function?
predictions.select().show(25)

```

Once the job execution is done, try evaluating on how we performed!


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))
```

### Extra: Use the confusion matrix to see the other metrics


```python
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

preds_and_labels = predictions.select(['prediction','encoded']).withColumn('encoded', F.col('encoded').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','encoded'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())
```
