import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

trait SparkSessionWrapper {
  lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("spark clouds")
      .getOrCreate()
  }

  private val conf = new SparkConf().setAppName("appName").setMaster("master")
  lazy  val sc = new SparkContext(conf)
}

trait Settings extends SparkSessionWrapper {

}
