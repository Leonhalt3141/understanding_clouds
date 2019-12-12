
package risk

import java.io.File
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

import org.apache.spark

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.stat.KernelDensity
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions
import org.apache.spark.util.StatCounter
import breeze.plot._
import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.stat.correlation.Covariance
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.scalatest.FunSpec


object Risk extends FunSpec with Settings {
  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val runRisk = new Risk(spark)

    val (stocksReturns, factorsReturns) = runRisk.readStocksAndFactors()
    println(stocksReturns.head.mkString)
    println(factorsReturns.length)
    println(factorsReturns(2).mkString)

    runRisk.plotDistribution(factorsReturns.head)
    runRisk.plotDistribution(factorsReturns(1))

    val numTrials = 1000000
    val parallelism = 1000
    val baseSeed = 1001L

//    val trials = runRisk.computeTrialReturns(stocksReturns, factorsReturns, baseSeed, numTrials,
//      parallelism)
  }
}

class Risk(private val spark: SparkSession) {
  import spark.implicits._

  def readGoogleHistory(file: File): Array[(LocalDate, Double)] = {
    val formatter = DateTimeFormatter.ofPattern("d-MMM-yy", Locale.ENGLISH)
    val lines = scala.io.Source.fromFile(file).getLines().toSeq
    lines.tail.map { line =>
      val cols = line.split(',')
      val date = LocalDate.parse(cols(0).mkString, formatter)
      val value = cols(4).toDouble
      (date, value)
    }.reverse.toArray
  }

  def trimToRegion(history: Array[(LocalDate, Double)], start: LocalDate, end: LocalDate)
  : Array[(LocalDate, Double)] = {
    var trimmed = history.dropWhile(_._1.isBefore(start)).
      takeWhile(x => x._1.isBefore(end) || x._1.isEqual(end))
    if (trimmed.head._1 != start) {
      trimmed = Array((start, trimmed.head._2)) ++ trimmed
    }
    if (trimmed.last._1 != end) {
      trimmed = trimmed ++ Array((end, trimmed.last._2))
    }
    trimmed
  }

  def fillInHistory(history: Array[(LocalDate, Double)], start: LocalDate, end: LocalDate)
  : Array[(LocalDate, Double)] = {
    var cur = history
    val filled = new ArrayBuffer[(LocalDate, Double)]()
    var curDate = start
    while (curDate.isBefore(end)) {
      if (cur.tail.nonEmpty && cur.tail.head._1 == curDate) {
        cur = cur.tail
      }

      filled += ((curDate, cur.head._2))

      curDate = curDate.plusDays(1)
      // Skip weekends
      if (curDate.getDayOfWeek.getValue > 5) {
        curDate = curDate.plusDays(2)
      }
    }
    filled.toArray
  }

  def twoWeekReturns(history: Array[(LocalDate, Double)]): Array[Double] = {
    history.sliding(10).map { window =>
      val next = window.last._2
      val prev = window.head._2
      (next - prev) / prev
    }.toArray
  }


  def readStocksAndFactors(): (Seq[Array[Double]], Seq[Array[Double]]) = {
    val start = LocalDate.of(2009, 10, 23)
    val end = LocalDate.of(2014, 10, 23)

    val stocksDir = new File("data/risk/stocks/")
    val files = stocksDir.listFiles()
    val allStocks = files.iterator.flatMap { file =>
      try {
        Some(readGoogleHistory(file))
      } catch {
        case e: Exception => None
      }
    }
    val rawStocks = allStocks.filter(_.length >= 260 * 5 + 10)

    val factorsPrefix = "data/risk/factors/"
    val rawFactors = Array("NYSEARCA3AGLD.csv", "NASDAQ3ATLT.csv", "NYSEARCA3ACRED.csv").
      map(x => new File(factorsPrefix + x)).
      map(readGoogleHistory)

    val stocks = rawStocks.map(trimToRegion(_, start, end)).map(fillInHistory(_, start, end))

    val factors = rawFactors.
      map(trimToRegion(_, start, end)).
      map(fillInHistory(_, start, end))

    val stocksReturns = stocks.map(twoWeekReturns).toArray.toSeq
    val factorsReturns = factors.map(twoWeekReturns)
    (stocksReturns, factorsReturns)

  }

  def factorMatrix(histories: Seq[Array[Double]]): Array[Array[Double]] = {
    val mat = new Array[Array[Double]](histories.head.length)
    for (i <- histories.head.indices) {
      mat(i) = histories.map(_(i)).toArray
    }
    mat
  }

  def featurize(factorReturns: Array[Double]): Array[Double] = {
    val squaredReturns = factorReturns.map(x => math.signum(x) * x * x)
    val squareRootedReturns = factorReturns.map(x => math.signum(x) * math.signum(x) * math.sqrt(math.abs(x)))
    squaredReturns ++ squareRootedReturns ++ factorReturns
  }

  def linearModel(instrument: Array[Double], factorMatrix: Array[Array[Double]])
    : OLSMultipleLinearRegression = {
    val regression = new OLSMultipleLinearRegression()
    regression.newSampleData(instrument, factorMatrix)
    regression
  }

  def trialReturns(
                  seed: Long,
                  numTrials: Int,
                  instruments: Seq[Array[Double]],
                  factorMeans: Array[Double],
                  factorCovariences: Array[Array[Double]]
                  ): Seq[Double] = {
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(rand, factorMeans, factorCovariences)

    val trialReturns = new Array[Double](numTrials)
    for (i <- 0 until numTrials) {
      val trialFactorReturns = multivariateNormal.sample()
      val trialFeatures = featurize(trialFactorReturns)
      trialReturns(i) = trialReturn(trialFeatures, instruments)
    }
    trialReturns
  }

  def trialReturn(trial: Array[Double], instruments: Seq[Array[Double]]): Double = {
    var totalReturn = 0.0
    for (instrument <- instruments) {
      totalReturn += instrumentTrialReturn(instrument, trial)
    }
    totalReturn / instruments.length
  }

  def instrumentTrialReturn(instrument: Array[Double], trial: Array[Double]): Double = {
    var instrumentTrialReturn = instrument(0)
    var i = 0
    while (i < trial.length) {
      instrumentTrialReturn += trial(i) * instrument(i+1)
      i += 1
    }
    instrumentTrialReturn
  }

  def plotDistribution(samples: Array[Double]): Figure = {
    val min = samples.min
    val max = samples.max
    val stddev = new StatCounter(samples).stdev
    val bandwidth = 1.06 * stddev * math.pow(samples.length, -.2)

    val domain = Range.BigDecimal(min, max, (max - min) / 100).map(_.toDouble).toList.toArray
    val kd = new KernelDensity().
      setSample(samples.toSeq.toDS.rdd).
      setBandwidth(bandwidth)

    val densities = kd.estimate(domain)
    val f = Figure()
    val p = f.subplot(0)
    p += plot(domain, densities)
    p.xlabel = "Two Week Return ($)"
    p.ylabel = "Density"
    f
  }

  def plotDistribution(samples: Dataset[Double]): Figure = {
    val (min, max, count, stddev) = samples.agg(
      functions.min($"value"),
      functions.max($"value"),
      functions.count($"value"),
      functions.stddev_pop($"value")
    ).as[(Double, Double, Long, Double)].first()
    val bandwidth = 1.06 * stddev * math.pow(count, -.2)

    // Using toList before toArray avoids a Scala bug
    val domain = Range.BigDecimal(min, max, (max - min) / 100).map(_.toDouble).toList.toArray
    val kd = new KernelDensity().
      setSample(samples.rdd).
      setBandwidth(bandwidth)
    val densities = kd.estimate(domain)
    val f = Figure()
    val p = f.subplot(0)
    p += plot(domain, densities)
    p.xlabel = "Two Week Return ($)"
    p.ylabel = "Density"
    f
  }

  def fivePercentVaR(trials: Dataset[Double]): Double = {
    val quantiles = trials.stat.approxQuantile("value", Array(0.05), 0.0)
    quantiles.head
  }

  def fivePercentCVaR(trials: Dataset[Double]): Double = {
    val topLosses = trials.orderBy("value").limit(math.max(trials.count().toInt / 20, 1))
    topLosses.agg("value" -> "agg").first()(0).asInstanceOf[Double]
  }

//
//  def computeTrialReturns(
//                         stocksReturns: Seq[Array[Double]],
//                         factorsReturns: Seq[Array[Double]],
//                         baseSeed: Long,
//                         numTrials: Int,
//                         parallelism: Int
//                         ): Dataset[Double] = {
//    val factorMat = factorMatrix(factorsReturns)
//    val factorCov = new Covariance(factorMat).getCovarianceMatrix().getData()
//    val factorMeans = factorsReturns.map(factor => factor.sum / factor.length).toArray
//    val factorFeatures = factorMat.map(featurize)
//    val factorWeights = computeFactorWeigths(stocksReturns, factorFeatures)
//
//    val seeds = (baseSeed until baseSeed + parallelism)
//    val seedDS = seeds.toDS().repartition(parallelism)
//
//    seedDS.flatMap(trialReturns(_, numTrials / parallelism, factorWeights, factorMeans, factorCov))
//  }


}