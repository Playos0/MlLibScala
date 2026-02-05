package ml.metrics


object Metrics {

  /**
   * Podaje Predykcje modeluw - pierwszy parametr
   * Podaje rzeczywiste wartosci
   * A on zwraca "celność"
   * */

  def r2(predictions: Vector[Double], targets: Vector[Double]): Double = {

    val meanY = targets.sum / targets.size
    val ssRes = predictions.zip(targets).map {case (p, t) => (t - p) * (t - p)}.sum
    val ssTot = targets.map(t => (t - meanY) * (t - meanY)).sum
    1 - (ssRes / ssTot)
  }
}