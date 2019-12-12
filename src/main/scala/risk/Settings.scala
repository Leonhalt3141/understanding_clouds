package risk

import org.apache.spark.sql.SparkSession


trait SparkSessionWrapper {
  lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("spark clouds")
      .getOrCreate()
  }


}

trait Settings extends SparkSessionWrapper {

}
