import scala.math.random
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, IDF, StopWordsRemover,VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, Transformer, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

/** Computes an approximation to pi */
object SparkPi {
  def main(args: Array[String]) {
    val conf:SparkConf = new SparkConf().setAppName("Yelp")
    val sc = new SparkContext(conf:SparkConf)
    val spark = SparkSession.builder.getOrCreate()
    val time = System.nanoTime
    val readin = spark.read.json("../../data/yelp_sample.json")
    println(readin.printSchema())
    val columns = Seq("review_id","text","label")
    val text_to_token = readin.select("review_id","text","stars").toDF(columns: _*)
    println(text_to_token.show())
    val Array(train, test) = text_to_token.randomSplit(Array(.8,.2),12345)


    val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

    val stopword = new StopWordsRemover()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("no_stops")

    val hashingTF = new HashingTF()
    .setInputCol(stopword.getOutputCol)
    .setOutputCol("hashing")

    val idf = new IDF()
    .setInputCol(hashingTF.getOutputCol)
    .setMinDocFreq(5)
    .setOutputCol("features")

    val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.001)

    val pipeline = new Pipeline()
    .setStages(Array(tokenizer, stopword,hashingTF,idf, lr))

    val lr_model = pipeline.fit(train)

    val lr_prediction = lr_model.transform(test)

    println(lr_prediction.printSchema())

    val metrics = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")

    val precision = metrics.setMetricName("weightedPrecision").evaluate(lr_prediction)
    val recall = metrics.setMetricName("weightedRecall").evaluate(lr_prediction)
    val f1Score = metrics.setMetricName("f1").evaluate(lr_prediction)
    val accuracy = metrics.setMetricName("accuracy").evaluate(lr_prediction)
    println("Summary Stats")
    println("Accuracy = " + accuracy)
    println("Precision = " + precision)
    println("Recall = " + recall)
    println("F1 Score = " + f1Score)

    val x = sc.parallelize(Seq(
        Seq(
        accuracy, lr_prediction.count(),precision,recall, f1Score,time, System.nanoTime,(System.nanoTime- time) / 1e9d
        ),
        Seq(
        "Accuracy","total","Precision","Recall","F1 Score","start","end", "total_time"
        )
      )
    )
    println(x.collect())
    spark.stop()
  }
}
