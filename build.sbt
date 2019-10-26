name := "understanding_clouds"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
//  jdbc , ehcache , ws , specs2 % Test , guice,
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta4",
//  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta4",
  "org.apache.spark" %% "spark-core" % "2.4.0",
  "org.apache.spark" %% "spark-mllib" % "2.4.0",
  "org.apache.spark" %% "spark-streaming" % "2.4.0",
  //  "com.typesafe.slick" %% "slick" % "3.3.1",
  "org.postgresql" % "postgresql" % "42.2.2",
  "net.postgis" % "postgis-jdbc" % "2.2.1",
  "org.scalatest" %% "scalatest" % "3.0.5",
  "com.softwaremill.sttp" %% "core" % "1.1.6"
)
