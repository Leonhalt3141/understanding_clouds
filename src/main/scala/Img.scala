
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.broadcast.Broadcast
import org.scalatest.FunSpec
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation._
import org.apache.spark.mllib.recommendation.{ALS => ALSM}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd._
import org.apache.spark.sql._

import scala.util.Random


class LastFMRead extends Settings {
  import spark.implicits._

  private val user_artist_path = "data/lastfm/user_artist_data.txt"
  private val artist_path = "data/lastfm/artist_data.txt"
  private val artist_alias_path = "data/lastfm/artist_alias.txt"

  val rawUserArtistData: Dataset[String] = spark.read.textFile(user_artist_path)
  val rawArtistData: Dataset[String] = spark.read.textFile(artist_path)
  val rawArtistAlias: Dataset[String] = spark.read.textFile(artist_alias_path)

  def readUserArtist(): DataFrame = {
    rawUserArtistData.map { line =>
      val Array(user, artist, score) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

  }

  def readArtist(): DataFrame = {
    rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }.toDF("id", "name")
  }

  def readArtistAlias(): DataFrame = {
    rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.toDF("psue_id", "id")
  }

  def makeArtistByIdDF(): DataFrame = {
    rawArtistData.flatMap{ line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      }else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case _: NumberFormatException => None
        }
      }
    }.toDF("id", "name")
  }

  def makeArtistById(data: RDD[String]): RDD[(Int, String)] = {
    data.flatMap{ line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      }else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case _: NumberFormatException => None
        }
      }
    }

  }

  def buildArtistAlias(rawArtistAlias: Dataset[String]): Map[Int,Int] = {
    rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap
  }

  def buildCounts (rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      (userID, finalArtistID, count)
    }.toDF("user", "artist", "count")
  }

  def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
    val toRecommend = model.itemFactors.select($"id".as("artist")).withColumn("user", lit(userID))
    toRecommend.cache()
    model.transform(toRecommend).select("artist", "prediction").
      orderBy($"prediction".desc).
      limit(howMany)
  }

  def areaUnderCurve(
                    positiveData: DataFrame,
                    bAllArtistIDs: Broadcast[Array[Int]],
                    predictFunction: DataFrame => DataFrame): Double = {

    val positivePredictions = predictFunction(positiveData.select("user", "artist")).
      withColumnRenamed("prediction", "positivePrediction")

    val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
      groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0
        // Make at most one pass over all artists to avoid an infinite loop.
        // Also stop when number of negative equals positive set size
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
          // Only add new distinct IDs
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // Return the set with user ID added back
        negative.map(artistID => (userID, artistID))
      }.toDF("user", "artist")

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user", "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user").as("correct")).
      select("user", "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
      select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }
}


object Img extends FunSpec with Settings {
  import spark.implicits._

  def withGoodVines()(df: DataFrame): DataFrame = {
    df.withColumn(
      "chi",
      lit("happy")
    )
  }

  def main(args: Array[String]): Unit = {
//    val df = List("sue", "fan").toDF("name")
//    val betterDF = df.transform(withGoodVines())
//    betterDF.show()
//
//    val df_ = spark.read.format("image").option("dropInvalid", true).load("data/train/")
//    df_.select("image.origin", "image.width", "image.height").show(truncate = false)


    val dataread = new LastFMRead()

    val userArtistDF = dataread.readUserArtist()

    userArtistDF.agg(min("user"), max("user"), min("artist"), max("artist")).show()

    val artistDF = dataread.readArtist()

    artistDF.show(10)
    println(artistDF.count())

    val artistAliasDF = dataread.readArtistAlias()

    artistAliasDF.show(10)
    println(artistAliasDF.count())

    artistDF.filter($"id" isin(1208690, 1003926, 6803336, 1000010)).show()


    val bArtistAlias = spark.sparkContext.broadcast(dataread.buildArtistAlias(dataread.rawArtistAlias))


    val trainData = dataread.buildCounts(dataread.rawUserArtistData, bArtistAlias)
//    trainData.cache()
    trainData.unpersist()



    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).
      setRegParam(0.01).
      setAlpha(1.0).
      setMaxIter(5).
      setUserCol("user").
      setItemCol("artist").
      setRatingCol("count").
      setPredictionCol("prediction").
      fit(trainData)

    model.userFactors.show(truncate = false)

    model.itemFactors.show(false)

    val userID = 2093760


    val topRecommendations = dataread.makeRecommendations(model, userID, 10)
    val recommendedArtistIDS = topRecommendations.select("artist").as[Int].collect()

    val artistByIdDF = dataread.makeArtistByIdDF()
    artistByIdDF.filter($"id".isin(recommendedArtistIDS:_*)).show()


    val existingArtistIDs = trainData.
      filter($"user" === userID).
      select("artist").as[Int].collect()


    val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
    artistByIdDF.join(spark.createDataset(recommendedArtistIDs).toDF("id"), "id").
      select("name").show()

    model.userFactors.unpersist()
    model.itemFactors.unpersist()

  }
}
