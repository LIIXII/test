package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{ColumnName, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object Preprocessor {



  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train/train_clean.csv")


    import df.sparkSession.implicits._
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))



    val df2: DataFrame = dfCasted.drop("disable_communication")

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")




    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")

    val dfFinal: DataFrame = dfCountry
      .withColumn("final_status", when($"final_status".isNull || $"final_status" =!= 1, 0).otherwise($"final_status"))

    dfFinal.groupBy("final_status").count.orderBy($"count".desc).show()

    val dfDate: DataFrame = dfFinal
      .withColumn("days_campaign", ($"deadline"-$"launched_at")/3600/24)
      .withColumn("hours_prepa", format_number((($"launched_at"-$"created_at")/3600),3))
      .drop("deadline","launched_at","created_at")
      .withColumn("hours_prepa",$"hours_prepa".cast("Double"))



    val dfText:DataFrame = dfDate
      .withColumn("name", lower($"name"))
      .withColumn("desc",lower($"desc"))
      .withColumn("keywords",lower($"keywords"))
      .withColumn("text",concat_ws(" ",$"name",$"desc",$"keywords"))
      .drop("name","desc","keywords")

    val dfNoNull: DataFrame = dfText
      .withColumn("days_campaign", when($"days_campaign".isNull,-1).otherwise($"days_campaign"))
      .withColumn("hours_prepa", when($"hours_prepa" .isNull,-1).otherwise($"hours_prepa"))
      .withColumn("goal", when($"goal".isNull,-1).otherwise($"goal"))
      .withColumn("currency2", when($"currency2".isNull,"unknown").otherwise($"currency2"))
      .withColumn("country2", when($"country2".isNull,"unknown").otherwise($"country2"))

    dfNoNull.show(10)

    dfNoNull.write.mode("overwrite").parquet("src/main/resources/preprocessed")
  }
}
