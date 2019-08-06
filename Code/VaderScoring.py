import re
import nltk
from pyspark import SparkConf, SparkContext
from pyspark import sql
from pyspark.sql import SQLContext
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from pyspark.sql.functions import col,udf
from pyspark.ml import Pipeline,Transformer
from pyspark.ml.classification import LogisticRegression,GBTClassifier
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover,StringIndexer,VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import time 
from pyspark.ml.util import keyword_only  
from pyspark.sql.types import DoubleType

start_time = time.time()
nltk.download('stopwords')
# Set of all stopwords 
swords = list(set(stopwords.words('english')))
conf = SparkConf().setAppName("Yelp")
sc = SparkContext(conf=conf)
spark = SQLContext(sc)
"""
Schema
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
loc = 'c' # 'c'
# Full
if loc == 'f':
    readin = spark.read.json("/data/MSA_8050_Spring_19/2pm_6/yelp_academic_dataset_review.json") 
    # There are 6685900 rows
elif loc == 'c':
# Sample
    readin = spark.read.json("/data/MSA_8050_Spring_19/2pm_6/yelp_academic_dataset_review_sample.json") 
    # There are 100 rows
# Local
else: 
    readin = sc.textFile("/data/MSA_8050_Spring_19/2pm_6/VaderScores") 
    
print readin.printSchema()
###############################################################################################
# Read in the data and only take the needed columns
vader = SentimentIntensityAnalyzer()
# turns DataFrame into RDD
text = readin.rdd.map(list)
# creates key value pair (review_id, [list of tokenized sentences])
text = text.map(lambda x: (x[4],tokenize.sent_tokenize(x[6])))
# flat maps it so now it is (review_id, tokenized sentence)
# So there will be duplicates of the key review_id with each sentence attached
# to said review
text = text.flatMapValues(lambda x: x)
# Map the sentence into the Sentiment analyzer to grab scores as a string
text = text.mapValues(lambda x: str(vader.polarity_scores(x)))
# The format of the vader score is '{"positive": ___, "neutral": ____, "negative": ____, "compound": ____}'
# Since it is a string, we can split by compound, get the ____}' and split again on } to get the number
# Then convert to float
text = text.mapValues(lambda x: float(x.split("compound': ")[1].split("}")[0]))
# Want to get the average sentiment of each sentence in that review
# Setup with (review_id, (score, 1))
text = text.mapValues(lambda x: (x,1))
# reduce by key by adding (score, 1)
text = text.reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1]))
# divide sum of score/number of sentences
text = text.mapValues(lambda x: x[0]/x[1])
text.saveAsTextFile('/data/MSA_8050_Spring_19/2pm_6/VaderScores_sample')