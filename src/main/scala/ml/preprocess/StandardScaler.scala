package ml.preprocess

import ml.math.Vec

class StandardScaler(val mean: Vec, val std: Vec) {

  def transform(x: Vec): Vec =
    (x - mean) / std

  def transformAll(xs: Vector[Vec]): Vector[Vec] =
    xs.map(transform)
}

object StandardScaler{

  def fit(xs: Vector[Vec]): StandardScaler ={
    val n = xs.size
    val numFeatures = xs.head.size

    val mean = xs.reduce(_ + _) * (1.0 / n)

    val variance =
      xs.map(x => (x - mean).map(v => v * v))
        .reduce(_ + _) * (1.0 / n)

    val std = variance.map(v => math.sqrt(v) + 1e-8)

    new StandardScaler(mean, std)
  }
}
