package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.IntegerType



object Preprocessor {

  def main(args: Array[String]): Unit = {

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

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/

      // Pour réarranger les quotes:

      // val csv_as_text = spark.read.text("/Users/remi/Desktop/Cours/Spark/train.csv")
      // val csv_as_text2=csv_as_text.withColumn("replaced", regexp_replace($"value", "\"\"+", " "))
      // csv_as_text2.select("replaced").write.text("/Users/remi/Desktop/Cours/Spark/train_clean.csv")


      var data = spark.read
        .option("header", true)
        .option("nullValue", "false")
        .csv("../train.csv")
        .toDF()
    data.show(20)


    // Get number of rows
    val nb_rows = data.count()
    println(nb_rows)
    //111792

    // Get number of columns (get column names and count arrray)
    println(data.columns.length)

    // Afficher   le   dataFrame   sous   forme   de   table.

    data.show(20)

    // Afficher le schema
    data.printSchema()

    // Assigner   le   type   “Int”   aux   colonnes   qui   vous   semblent   contenir   des   entiers.
    data = data.withColumn("goal", 'goal.cast(IntegerType))
      .withColumn("backers_count", 'backers_count.cast(IntegerType))
      .withColumn("final_status", 'final_status.cast(IntegerType))
    data.printSchema()





    /** 2 - CLEANING **/
    // Afficher une description statistique des colonnes de type Int (avec .describe().show   )

    data.describe("goal", "backers_count", "final_status").show()

    // Observer les autres colonnes, et proposer des cleanings à faire sur les données:
    // faites des groupBy count, des show, des dropDuplicates. Quels cleaning faire pour chaque colonne ?
    // Y a-t-il des colonnes inutiles ? Comment traiter les valeurs manquantes ?
    // Des “fuites du futur” ???

    import org.apache.spark.sql.functions._
    data.groupBy("disable_communication").count().sort(desc("count")).show()

    // Drop la colonne disable_communication
    data = data.drop("disable_communication")
    println(data.columns.length)

    // D. Enlever la colonne backers_count et state_changed_at
    data = data.drop("backers_count")
    data = data.drop("state_changed_at")
    println(data.columns.length)

    // e)
    // On pourrait penser que "currency" et "country" sont redondantes, auquel cas on pourrait enlever une des colonne.
    // Mais en y regardant de plus près:
    //   - dans la zone euro: même monnaie pour différents pays => garder les deux colonnes.
    //   - il semble y avoir des inversions entre ces deux colonnes et du nettoyage à faire en utilisant les deux colonnes.
    //     En particulier on peut remarquer que quand country=false le country à l'air d'être dans currency:

    data.filter($"country".isNull).groupBy("currency").count.orderBy($"count".desc).show(50)

    def udf_country = udf{(country: String, currency: String) =>
      if (country == null) // && currency != "false")
        currency
      else
        country //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    def udf_currency = udf{(currency: String) =>
      if ( currency != null && currency.length != 3 )
        null
      else
        currency //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    val dfCountry: DataFrame = data
      .withColumn("country2", udf_country($"country", $"currency"))
      .withColumn("currency2", udf_currency($"currency"))
      .drop("country", "currency")

    dfCountry.groupBy("country2", "currency2").count.orderBy($"count".desc).show(50)

    // Pour aider notre algorithme, on souhaite qu'un même mot écrit en minuscules ou majuscules ne soit pas deux
    // "entités" différentes. On met tout en minuscules
    val dfLower: DataFrame = dfCountry
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    dfLower.show(50)


    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/

    // a) b) c) features à partir des timestamp
    val dfDurations: DataFrame = dfLower
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2")) // datediff requires a dateType
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600.0, 3)) // here timestamps are in seconds, there are 3600 seconds in one hour
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)
      .drop("created_at", "deadline", "launched_at")


    // d)
    val dfText= dfDurations
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    /** VALEUR NULLES **/

    val dfReady: DataFrame = dfText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1
      ))

    // vérifier l'équilibrage pour la classification
    dfReady.groupBy("final_status").count.orderBy($"count".desc).show

    // filtrer les classes qui nous intéressent
    // Final status contient d'autres états que Failed ou Succeed. On ne sait pas ce que sont ces états,
    // on peut les enlever ou les considérer comme Failed également. Seul "null" est ambigue et on les enlève.
    val dfFiltered = dfReady.filter($"final_status".isin(0, 1))

    dfFiltered.show(50)
    println(dfFiltered.count)


    /** WRITING DATAFRAME **/

    //dfFiltered.write.mode(SaveMode.Overwrite).parquet("/Users/remi/Desktop/Cours/Scala/TP_parisTech_2017_2018/data/prepared_trainingset")



  }

}
