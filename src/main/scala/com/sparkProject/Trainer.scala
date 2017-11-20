package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


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
      .appName("TP_spark")
      .getOrCreate()

    /* Réduction des outputs en console pour qu'il n'affiche que les messages de niveau warning minimum */
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   val raw_data = spark.read.parquet("./data/prepared_trainingset")
    raw_data.show()
    println("Nombre de colonnes:")
    println(raw_data.columns.length)
    println("Nombre de lignes")
    println(raw_data.count())


    /** TF-IDF **/
      //Stage 1: Tokenizer
      // Séparation des textes en mots.
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stage 2: StopWordsRemover
    // On enlève les stopwords, les mots n'ayant pas de sens pour l'analyse.
    StopWordsRemover.loadDefaultStopWords("english")
    val stop_words_remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("tokens_without_stopwords")

    // Stage 3: Partie TF du TFIDF
    // On va utiliser la fréquence des mots

    val cvm = new CountVectorizer()
      .setInputCol(stop_words_remover.getOutputCol)
      .setOutputCol("tf")

    // Stage 4: Partie IDF du TFIDF
    val idf = new IDF()
      .setInputCol(cvm.getOutputCol)
      .setOutputCol("tfidf")

    /** VECTOR ASSEMBLER **/

    //Stage 5: On converti la colonne country en données numériques
    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
    // One hot encoder (ajout personel) on drop la derniere colonne pour décorréler
    val country_encoder = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("countryVec")
      .setDropLast(true)

    //Stage 6: On converti la colonne currency en données numériques
    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
    // One hot encoder (ajout personel) on drop la derniere colonne pour décorréler
    val currency_encoder = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currencyVec")
      .setDropLast(true)

    //Stage 7: On assemble les colonnes précédentes dans une seule pour appliquer l'algorithme
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign",
        "hours_prepa", "goal", "countryVec", "currencyVec"))
      .setOutputCol("features")



    /** MODEL **/
    //Stage 8: Application de la Regression logistique

    val lr  = new LogisticRegression()
      .setElasticNetParam( 0.0 )
      .setFitIntercept( true )
      .setFeaturesCol( "features" )
      .setLabelCol( "final_status" )
      .setStandardization( true )
      .setPredictionCol( "predictions" )
      .setRawPredictionCol( "raw_predictions" )
      .setThresholds( Array ( 0.7 ,  0.3 ))
      .setTol( 1.0e-6 )
      .setMaxIter( 300 )


    /** PIPELINE **/
    // Construction du pipeline:

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,
        stop_words_remover,
        cvm,
        idf,
        country_indexer,
        country_encoder,
        currency_indexer,
        currency_encoder,
        assembler,
        lr))

    /** TRAINING AND GRID-SEARCH **/
    // Train test split:

    val Array(training, test) = raw_data.randomSplit(Array(0.9, 0.1))

    // Grid search Sur les paramètes minDF et regParam du modèle
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvm.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      //.addGrid(pca.k, Array(3, 5, 7))
      .build()

    // On évalue le modèle avec une métrique f1 utile pour avoir le recall et la précision
    val myevaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // On créé le modele du grid_search avec cross-validation
    val grid_model = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(myevaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // On lance la cross-validation et choisi le meilleur jeux de paramètres.

    val TrainValidationModel = grid_model.fit(training)
    print(TrainValidationModel.bestModel)


    /** Predictions **/
      // On créée un nouveau dataframe contenant le resultat du meilleur modèle issu de la grid search
      // Appliqué sur le jeu de données test.
    val df_WithPredictions = TrainValidationModel.bestModel.transform(test).toDF()

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    // On calcule et affiche le score final
    val final_score = myevaluator.evaluate(df_WithPredictions)
    print("-----FINAL SCORE-------- : ", final_score)


    // On sauvegarde le modele final, issu de notre gridsearch.
    TrainValidationModel.write.overwrite.save("gridsearch_pipeline_tfidf")



  }
}
