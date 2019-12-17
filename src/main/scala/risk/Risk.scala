
package risk

import java.io.File

import scala.io.Source
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.stat.KernelDensity
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, functions}
import org.apache.spark.util.StatCounter
import breeze.plot._
import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.stat.correlation.Covariance
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.scalatest.FunSpec


object Risk extends FunSpec with Settings {

  def main(args: Array[String]): Unit = {
    val runRisk = new Risk(spark)

    val (stocksReturns, factorsReturns) = runRisk.readStocksAndFactors()
    println(stocksReturns.head.mkString)
    println(factorsReturns.length)
    println(factorsReturns(2).mkString)

    println("Stock data count: ", runRisk.rawStockData.count())
    println(runRisk.rawStockData.show(10))
    println(runRisk.rawStockData.take(0).toString.split(',')(0).mkString)
    println(runRisk.rawStockData.take(0).toString.split(',')(0) == "Date")
    println("Factor data count", runRisk.rawFactorData.count())

    val stockDF = runRisk.readStockData()
    println(stockDF.show(10))

    runRisk.plotDistribution(factorsReturns.head)
    runRisk.plotDistribution(factorsReturns(1))

    val numTrials = 10000000
    val parallelism = 10
    val baseSeed = 1001L

    val trials = runRisk.computeTrialReturns(stocksReturns, factorsReturns, baseSeed, numTrials,
      parallelism)
    trials.cache()
    val tbroadcast = spark.sparkContext.broadcast(trials)
//    trials.unpersist()
    println(trials)
    tbroadcast.unpersist()
    println(trials.count())

    val valueAtRisk = runRisk.fivePercentVaR(trials)
    val conditionalValueAtRisk = runRisk.fivePercentCVaR(trials)
    println("VaR 5%: " + valueAtRisk)
    println("CVaR 5%: " + conditionalValueAtRisk)

    val varConfidenceInterval = runRisk.bootstrappedConfidenceInterval(trials,
      runRisk.fivePercentVaR, 100, .05)
    val cvarConfidenceInterval = runRisk.bootstrappedConfidenceInterval(trials,
      runRisk.fivePercentCVaR, 100, .05)
    println("VaR confidence interval: " + varConfidenceInterval)
    println("CVaR confidence interval: " + cvarConfidenceInterval)
//    println("Kupiec test p-value: " + runRisk.kupiecTestPValue(stocksReturns, valueAtRisk, 0.05))
    runRisk.plotDistribution(trials)
  }
}

class Risk(private val spark: SparkSession) {
  import spark.implicits._

  private val stock_dir = "data/risk/stocks/"
  private val factor_dir = "data/risk/factors/"

  val rawStockData: Dataset[String] = spark.read.textFile(stock_dir)
  val rawFactorData: Dataset[String] = spark.read.textFile(factor_dir)

  def readStockData(): DataFrame = {
    val formatter = DateTimeFormatter.ofPattern("d-MMM-yy", Locale.ENGLISH)
    rawStockData.flatMap {line =>
      val Array(date, open, high, low, close, volume) = line.split(",")
      try {
        if (date matches "Date") {
          None
        } else {
          Some((date, open.toDouble, high.toDouble,
            low.toDouble, close.toDouble, volume.toInt))
        }
      } catch {
        case _: NumberFormatException => None
      }
    }.toDF("date", "open", "high", "low", "close", "volume")
  }

  def readGoogleHistory(file: File): Array[(LocalDate, Double)] = {
    val formatter = DateTimeFormatter.ofPattern("d-MMM-yy", Locale.ENGLISH)
    val source = Source.fromFile(file)
    val lines = source.getLines().toSeq
    val results = lines.tail.map { line =>
      val cols = line.split(',')
      val date = LocalDate.parse(cols(0), formatter)
      val value = cols(4).toDouble
      (date, value)
    }.reverse.toArray
    source.close()
    results
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
    val squareRootedReturns = factorReturns.map(x => math.signum(x) * math.sqrt(math.abs(x)))
    squaredReturns ++ squareRootedReturns ++ factorReturns
  }

  def linearModel(instrument: Array[Double], factorMatrix: Array[Array[Double]])
    : OLSMultipleLinearRegression = {
    val regression = new OLSMultipleLinearRegression()
    regression.newSampleData(instrument, factorMatrix)
    regression
  }

  def computeFactorWeights(
                            stocksReturns: Seq[Array[Double]],
                            factorFeatures: Array[Array[Double]]): Array[Array[Double]] = {
    stocksReturns.map(linearModel(_, factorFeatures)).map(_.estimateRegressionParameters()).toArray
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
    totalReturn / instruments.size
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

  def bootstrappedConfidenceInterval(
                                    trials: Dataset[Double],
                                    computeStatistic: Dataset[Double] => Double,
                                    numResamples: Int,
                                    probability: Double
                                    ): (Double, Double) = {
    val replacement = true
    val stats = (0 until numResamples).map{ _ =>
      val resample = trials.sample(replacement, 1.0)
      computeStatistic(resample)
    }.sorted
    val lowerIndex = (numResamples * probability / 2 - 1).toInt
    val upperIndex = math.ceil(numResamples * (1 - probability / 2)).toInt
    (stats(lowerIndex), stats(upperIndex))
  }

  def countFailures(stocksReturns: Seq[Array[Double]], valueAtRisk: Double): Int = {
    var failures = 0
    for (i <- stocksReturns.head.indices) {
      val loss = stocksReturns.map(_(i)).sum
      if (loss < valueAtRisk) {
        failures += 1
      }
    }
    failures
  }

  def kupiecTestStatistic(total: Int, failures: Int, confidenceLevel: Double): Double = {
    val failureRatio = failures.toDouble / total
    val logNumer = (total - failures) * math.log1p(-confidenceLevel) * failures * math.log(confidenceLevel)

    val logDenom = (total - failures) * math.log1p(-failureRatio) * failures * math.log(failureRatio)
    -2 * (logNumer - logDenom)
  }

  def kupiecTestPValue(
                      stocksReturns: Seq[Array[Double]],
                      valueAtRisk: Double,
                      confidenceLevel: Double
                      ): Double = {
    val failures = countFailures(stocksReturns, valueAtRisk)
    val total = stocksReturns.head.length
    val testStatistic = kupiecTestStatistic(total, failures, confidenceLevel)
    1 - new ChiSquaredDistribution(1.0).cumulativeProbability(testStatistic)
  }



  def computeTrialReturns(
                           stocksReturns: Seq[Array[Double]],
                           factorsReturns: Seq[Array[Double]],
                           baseSeed: Long,
                           numTrials: Int,
                           parallelism: Int): Dataset[Double] = {
    val factorMat = factorMatrix(factorsReturns)
    val factorCov = new Covariance(factorMat).getCovarianceMatrix.getData
    val factorMeans = factorsReturns.map(factor => factor.sum / factor.length).toArray
    val factorFeatures = factorMat.map(featurize)
    val factorWeights = computeFactorWeights(stocksReturns, factorFeatures)

    // Generate different seeds so that our simulations don't all end up with the same results
    val seeds = baseSeed until baseSeed + parallelism
    val seedDS = seeds.toDS().repartition(parallelism)

    // Main computation: run simulations and compute aggregate return for each
    seedDS.flatMap(trialReturns(_, numTrials / parallelism, factorWeights, factorMeans, factorCov))
  }


}