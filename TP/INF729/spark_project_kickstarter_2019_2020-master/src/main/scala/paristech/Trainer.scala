package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoder, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    val preprocessedDf: DataFrame = spark.read.parquet("src/main/resources/preprocessed")
    //Utilisation des données textuelles
    //Stage 1 : récupérer les mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val regexTokenizedDF = tokenizer.transform(preprocessedDf)

    //Stage 2 : retirer les stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    //val filteredDF = remover.transform(regexTokenizedDF)
    //CountVectorizerModel
    //Stage 3 : computer la partie TF
    val cvModel  = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setVocabSize(100)
      .setMinDF(2)
      //.fit(filteredDF)

    //val rawFeaturedDF = cvModel.transform(filteredDF)

    //Stage 4 : computer la partie IDF
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("tfidf")
      //.fit(rawFeaturedDF)

    //val rescaledDf = idf.transform(rawFeaturedDF)

    //Conversion des variables catégorielles en variables numériques
    //Stage 5 : convertir country2 en quantités numériques

    val stringIndexer1 =  new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")
      //.fit(rescaledDf)

    //val countryIndexedDf = stringIndexer1.transform(rescaledDf)

    //Stage 6 : convertir currency2 en quantités numériques
    val stringIndexer2 =  new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")
      //.fit(countryIndexedDf)

    //val indexedDf = stringIndexer2.transform(countryIndexedDf)

    //Stages 7 et 8: One-Hot encoder ces deux catégories

    val oneHotEncoder1 =  new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed"))
      .setOutputCols(Array("country_onehot"))
      //.fit(indexedDf)

    //val oneHotcountryDF = oneHotEncoder1.transform(indexedDf)

    val oneHotEncoder2 =  new OneHotEncoderEstimator()
      .setInputCols(Array("currency_indexed"))
      .setOutputCols(Array("currency_onehot"))
      //.fit(oneHotcountryDF)

    //val oneHotDF = oneHotEncoder2.transform(oneHotcountryDF)


    //Stage 9 : assembler tous les features en un unique vecteur
    //import oneHotDF.sparkSession.implicits._

    val assembled = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    //val assembledDf = assembled.transform(oneHotDF)

    //Stage 10 : créer/instancier le modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel,idf,stringIndexer1,stringIndexer2,oneHotEncoder1,oneHotEncoder2,assembled,lr))

    val Array(training, test) = preprocessedDf.randomSplit(Array(0.9, 0.1), seed = 12345)

    val model = pipeline.fit(training)

    val dfWithSimplePredictions = model.transform(test)

    val evalution = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setMetricName("f1")
      .setPredictionCol("predictions")


    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF,Array(55.0,75.0,95.0))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evalution)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)
      // Evaluate up to 2 parameter settings in parallel
      .setParallelism(2)


    val model_grid = trainValidationSplit.fit(training)

    val dfWithGridPredictions = model_grid.transform(test)



    println("\n")
    println("f1 score Test WithSimplePredictions = "+evalution.evaluate(dfWithSimplePredictions))
    println("\n")

    println("\n")
    println("f1 score Test WithGridPredictions = "+evalution.evaluate(dfWithGridPredictions))
    println("\n")




















  }
}
