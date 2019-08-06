
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.ml import PipelineModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize 
vader = SentimentIntensityAnalyzer()
from pyspark.sql import Row, SparkSession
import sys
# Command: 
# spark-submit Streaming_Model.py

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']
if __name__ == "__main__":
	# Spark Context
	sc = SparkContext("local[2]",appName = "StreamingReviews")
	sc.setLogLevel("ERROR")
	# Update Stream every 10 seconds
	ssc = StreamingContext(sc,10)
	# Load Model 
	lr_model = PipelineModel.load('./Model')
	#Create DStream from data source
	lines = ssc.textFileStream('./Test')
	#Transformations and actions on DStream
	text = lines.map(lambda x: x[1:-1])
	def process(time, rdd):
		print("========= %s =========" % str(time))
		try:
			# Get the singleton instance of SparkSession
			spark = getSparkSessionInstance(rdd.context.getConf())
			# Remove Header
			head = rdd.first()
			rdd = rdd.filter(lambda x: x != head)
			# Convert RDD[String] to RDD[Row] to DataFrame
			rowRdd = rdd.map(lambda w: Row(text=w.encode('utf-8')))
			# Create new Data Frame
			streamed_data = spark.createDataFrame(rowRdd)
			# feed review into Pipeline
			lr_prediction = lr_model.transform(streamed_data)
			# Print out the Review and prediction
			lr_prediction.select('text','prediction').show()
		except:
			# if no data is passed, it will go here
			pass
	# Apply the process function to each RDD
	text.foreachRDD(process)
	#Start listening to the server
	ssc.start()
	ssc.awaitTermination()
