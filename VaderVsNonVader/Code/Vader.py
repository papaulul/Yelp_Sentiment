import re
import nltk
from pyspark import SparkConf, SparkContext
#from pyspark.sql.session import SparkSession
from pyspark import sql
from pyspark.sql import SQLContext
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from pyspark.ml import Pipeline,Transformer
from pyspark.ml.classification import LogisticRegression,GBTClassifier
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover,StringIndexer,VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import time 
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

###
# What to type: 
# spark-submit --master yarn --deploy-mode client --py-files nltk.zip FinalProject.py 
###

# This is the timer 
start_time = time.time()
# Makes sure we have the stop words
nltk.download('stopwords')
# Set of all stopwords 
swords = list(set(stopwords.words('english')))
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
# Set to f, c, or anything else 
location = 'f' # 'c'
# Full Data set
# There are 6685900 rows
if location == 'f':
    readin = spark.read.json("/data/MSA_8050_Spring_19/2pm_6/yelp_academic_dataset_review.json") 
    text = sc.textFile("/data/MSA_8050_Spring_19/2pm_6/VaderScores")
# Sample
# There are 100 rows
elif location == 'c':
    readin = spark.read.json("/data/MSA_8050_Spring_19/2pm_6/yelp_academic_dataset_review_sample.json") 
    text = sc.textFile("/data/MSA_8050_Spring_19/2pm_6/VaderScores_sample")
# Local
# There are 100 rows
else: 
    readin = spark.read.json("yelp_academic_dataset_review_sample.json") 
def binary_lablel(x):
    """
    Logistic can only do Binary in Spark 1.6.2. 
    In Spark 2.2+ it does support multiclass logistic
    """
    # Positive Review
    if x >3:
        return 1.0
    # Negative Review
    else:
        return 0.0

# Prints the Schema for the read in 
print(readin.printSchema())
###############################################################################################
# Read in the data and only take the needed columns
###############################################################################################
# Will only take certain columns 
text_to_token = readin.select('review_id','text', 'stars')
# Convert Vader Scores to DF
text = text.map(lambda x: [x[1:-1].split(",")[0][2:-1].strip(), float(x[1:-1].split(",")[1].strip()) ])

text = text.toDF(['review_id','vader'])

# Checking how each DF look 
print(text.show())
print(text_to_token.show())
print(text.printSchema())
print(text_to_token.printSchema())

# Join both DF on review_id
text_to_token = text_to_token.join(text, 'review_id' )

# convert "stars" to a binary variable Spark 1.6
#text_to_token = text_to_token.rdd.map(lambda x: [x[0],x[1],binary_lablel(x[2]), x[3]]).toDF(['review_id','text','label','vader'])

# gets the right data frame only for Spark 2.2 
text_to_token = text_to_token.rdd.map(lambda x: [x[0],x[1],x[2], x[3]]).toDF(['review_id','text','label','vader'])

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
stopword = StopWordsRemover(inputCol = tokenizer.getOutputCol(), outputCol = "no_stops", stopWords= swords)
# The cheaper way to do TF-IDF 
# Creates a hash that contains the term frequency 
# This mean there are no pairs with the value 0 
# It'll output: (number_of_words {index_from_previous: value, ...}) with no value = 0 
# If the value is 0, the index_from_previous will skip so there can be key that go 
# 0, 1, 6, 8, ... etc all based on the contents of the previous step
hashingTF = HashingTF(inputCol= stopword.getOutputCol(), outputCol="hashing")
# Performs the IDF part in TF-IDF 
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features1", minDocFreq=5) 
# Appends output Token-Stopwords-HashingTF-IDF with output of Vader
assembler = VectorAssembler(
    inputCols=["features1", "vader"],
    outputCol="features")
# Initialize Logistic Regression 
lr = LogisticRegression(maxIter=10, regParam=0.001)
# Creates pipeline 
pipeline = Pipeline(stages=[ tokenizer, stopword,hashingTF,idf, assembler, lr])

###############################################################################################
# Fit model to training set

#lr_model = PipelineModel.load('./ModelTest')
lr_model = pipeline.fit(train)
# Make predictions on test set
lr_prediction = lr_model.transform(test)
# Schema of prediction outcome
print(lr_prediction.printSchema())

lr_model.save('./Model_Vader_binary')


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
print(metrics.confusionMatrix().toArray())
print("Total Time: ",time.time()-start_time)
# Parallelize all information: tp, tn, fp,fn,total, recall, percision, start time, end time, total time 
# This will be the output we'll be looking for.
x = sc.parallelize(
    [(accuracy, lr_prediction.count(),precision,recall, f1Score,start_time, time.time(),time.time()- start_time),('Accuracy','total','Precision','Recall','F1 Score','start','end', 'total_time')]
    )
# Vader Results
x.saveAsTextFile('/data/MSA_8050_Spring_19/2pm_6/Vader_Results_binary')
print(x.collect())

# Non Vader Results
# x.saveAsTextFile('/data/MSA_8050_Spring_19/2pm_6/Normal_Results')
