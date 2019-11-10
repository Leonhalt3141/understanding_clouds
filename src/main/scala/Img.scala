
import org.apache.spark.broadcast.Broadcast
import org.scalatest.FunSpec
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation._
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
    val userArtistDF = rawUserArtistData.map { line =>
      val Array(user, artist, score) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

    userArtistDF
  }

  def readArtist(): DataFrame = {
    val artistDF = rawArtistData.flatMap { line =>
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

    artistDF
  }

  def readArtistAlias(): DataFrame = {
    val artistAliasDF = rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.toDF("psue_id", "id")

    artistAliasDF
  }

  def makeArtistById(): DataFrame = {
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

    val artistAlias = dataread.rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap

    val artistAliasDF = dataread.readArtistAlias()

    artistAliasDF.show(10)
    println(artistAliasDF.count())

    artistDF.filter($"id" isin(1208690, 1003926, 6803336, 1000010)).show()

    val bArtistAlias = spark.sparkContext.broadcast(artistAlias)

    val trainData = dataread.buildCounts(dataread.rawUserArtistData, bArtistAlias)
    trainData.cache()

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

    val toRecommend = model.itemFactors.select($"id".as("artist")).withColumn("user", lit(userID))
    toRecommend.cache()

    model.transform(toRecommend).orderBy($"prediction".desc).limit(15).show()

    val topRecommendations = dataread.makeRecommendations(model, userID, 10)
    val recommendedArtistIDS = topRecommendations.select("artist").as[Int].collect()

    val artistById = dataread.makeArtistById()
    artistById.filter($"id".isin(recommendedArtistIDS:_*)).show()

//    artistDF.withColumn("")

//   val transform = model.transform(trainData).toDF("user", "artist", "count", "prediction")
//    transform.withColumn("user", lit(userID)).withColumn("count", lit(0)).orderBy($"prediction".desc).limit(15).show()
//    transform.withColumn("user", lit(userID)).withColumn("artist", lit(2814)).show()

  }
}
