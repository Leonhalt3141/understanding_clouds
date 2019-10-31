
import org.scalatest.FunSpec
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._



object Img extends FunSpec with Settings {
  import spark.implicits._

  def withGoodVines()(df: DataFrame): DataFrame = {
    df.withColumn(
      "chi",
      lit("happy")
    )
  }

  def main(args: Array[String]): Unit = {
    val df = List("sue", "fan").toDF("name")
    val betterDF = df.transform(withGoodVines())
    betterDF.show()

    val df_ = spark.read.format("image").option("dropInvalid", true).load("data/train/")
    df_.select("image.origin", "image.width", "image.height").show(truncate = false)

//    val rawUserArtistData = spark.read.textFile("hdfs:///Users/kenkuwata/workspace/kaggle/understanding_clouds/data/lastfm/user_artist_data.txt")
    val rawUserArtistData = spark.read.textFile("data/lastfm/user_artist_data.txt")

    val userArtistDF = rawUserArtistData.map { line =>
      val Array(user, artist, score) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

    userArtistDF.agg(min("user"), max("user"), min("artist"), max("artist")).show()

    val rawArtistData = spark.read.textFile("data/lastfm/artist_data.txt")
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
    artistDF.show(10)
    println(artistDF.count())
  }
}
