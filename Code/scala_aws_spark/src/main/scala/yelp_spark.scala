import scala.math.random
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, IDF, StopWordsRemover,VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, Transformer, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.udf

/** Computes an approximation to pi */
object yelp_spark {
  def stars_sent(x: Double): Int = {
    if (x>3)
      return 3
    else{
      if (x<3)
        return 1 
      else 
        return 2
    }
  }

  def main(args: Array[String]) {
    val time = System.nanoTime
    val conf:SparkConf = new SparkConf().setAppName("Yelp")
    val sc = new SparkContext(conf:SparkConf)
    val spark = SparkSession.builder.getOrCreate()
    //val readin = spark.read.json("s3://yelp-spark-project/yelp_academic_dataset_review_sample.json")
    val readin = spark.read.json("s3://yelp-spark-project/yelp_academic_dataset_review.json")
    println(readin.printSchema())
    val columns = Seq("review_id","text","label")
    val text_to_token_col = readin.select("review_id","text","stars").toDF(columns: _*)
    val text_to_tokens_udf = udf(stars_sent _)
    import spark.implicits._

    val text_to_token = text_to_token_col.withColumn("label",text_to_tokens_udf($"label"))
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
        accuracy, lr_prediction.count(),precision,recall, f1Score,time, System.nanoTime,(System.nanoTime- time) / 1e9d)
        ,
      Seq(
        "Accuracy","total","Precision","Recall","F1 Score","start","end", "total_time")
      )
    )
    x.collect().foreach(println)
    x.saveAsTextFile("s3://yelp-spark-project/output/scala_full_2")
    spark.stop()
  }
}
