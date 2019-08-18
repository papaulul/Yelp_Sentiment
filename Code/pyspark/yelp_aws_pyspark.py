from pyspark import SparkConf, SparkContext
from pyspark import sql
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline,Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover,VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import time 
from pyspark.ml import PipelineModel
# This is the timer 
start_time = time.time()

"""
def stars_sent(x):
    if x < 3: 
        return 1
    elif x > 3: 
        return 3
    else: return 2
"""
# Makes sure we have the stop words
# Set of all stopwords 
# Setting up spark context 
conf = SparkConf().setAppName("Yelp")
sc = SparkContext(conf=conf)
# SQL Context so we can read the JSON files
spark = SQLContext(sc)
"""
Schema - Readin 
 |-- business_id: string (nullable = true)
 |-- cool: long (nullable = true)
 |-- date: string (nullable = true)
 |-- funny: long (nullable = true)
 |-- review_id: string (nullable = true)
 |-- stars: double (nullable = true)
 |-- text: string (nullable = true)
 |-- useful: long (nullable = true)
 |-- user_id: string (nullable = true)
"""
#readin = spark.read.json("s3://yelp-spark-project/yelp_academic_dataset_review_sample.json") 
readin = spark.read.json("s3://yelp-spark-project/yelp_academic_dataset_review.json") 
# Local
# There are 100 rows
# Prints the Schema for the read in 
print(readin.printSchema())
###############################################################################################
# Read in the data and only take the needed columns
###############################################################################################
# Will only take certain columns 
text_to_token = readin.select('review_id','text', 'stars')
# Convert Vader Scores to DF
# Checking how each DF look 
print(text_to_token.show())
print(text_to_token.printSchema())

# gets the right data frame only for Spark 2.2 
#text_to_token = text_to_token.rdd.map(lambda x: [x[0],x[1],stars_sent(x[2])]).toDF(['review_id','text','label'])
text_to_token = text_to_token.rdd.map(lambda x: [x[0],x[1],x[2]]).toDF(['review_id','text','label'])

# convert "stars" to a binary variable Spark 1.6
#text_to_token = text_to_token.rdd.map(lambda x: [x[0],x[1],binary_lablel(x[2])]).toDF(['review_id','text','label'])

print(text_to_token.show(),"\n")
###############################################################################################
#### Logistic Regression Pipeline ####
###############################################################################################
# Train Test Split
train, test = text_to_token.randomSplit([0.8, 0.2], seed=12345)
###############################################################################################
# Pipeline
###############################################################################################
# Tokenize by word 
tokenizer = Tokenizer(inputCol="text", outputCol="words")
# Remove stop words in the text
stopword = StopWordsRemover(inputCol = tokenizer.getOutputCol(), outputCol = "no_stops")
# The cheaper way to do TF-IDF 
# Creates a hash that contains the term frequency 
# This mean there are no pairs with the value 0 
# It'll output: (number_of_words {index_from_previous: value, ...}) with no value = 0 
# If the value is 0, the index_from_previous will skip so there can be key that go 
# 0, 1, 6, 8, ... etc all based on the contents of the previous step
hashingTF = HashingTF(inputCol= stopword.getOutputCol(), outputCol="hashing")
# Performs the IDF part in TF-IDF 
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features", minDocFreq=5) 
# Appends output Token-Stopwords-HashingTF-IDF with output of Vader
# Initialize Logistic Regression 
lr = LogisticRegression(maxIter=10, regParam=0.001)
# Creates pipeline 
pipeline = Pipeline(stages=[ tokenizer, stopword,hashingTF,idf, lr])

###############################################################################################
# Fit model to training set

#lr_model = PipelineModel.load('./ModelTest')
lr_model = pipeline.fit(train)
# Make predictions on test set
lr_prediction = lr_model.transform(test)
# Schema of prediction outcome
print(lr_prediction.printSchema())

#lr_model.save('./Model_binary')


lr_prediction = lr_prediction.select("prediction","label").rdd.map(lambda x: (x[0],x[1]))
metrics = MulticlassMetrics(lr_prediction)
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
accuracy = metrics.accuracy
print("Summary Stats")
print("Accuracy = %s" % accuracy)
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

# Parallelize all information: tp, tn, fp,fn,total, recall, percision, start time, end time, total time 
# This will be the output we'll be looking for.
x = sc.parallelize(
    [(accuracy, lr_prediction.count(),precision,recall, f1Score,start_time, time.time(),time.time()- start_time),('Accuracy','total','Precision','Recall','F1 Score','start','end', 'total_time')]
    )

print(x.collect())
x.saveAsTextFile('s3://yelp-spark-project/output/python_full_1')

