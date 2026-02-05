package ml.model

object Loss {
  //Błąd średniokwadratowy

  def mse(predictions: Vector[Double], labels: Vector[Double]): Double = {
    require(predictions.length == labels.length, "Predykcje i labele muszą mieć tą samą długość")

    val squaredErrors = predictions.zip(labels).map {case (yHat, y) =>
      val error = yHat - y
      error * error
    }

    squaredErrors.sum / predictions.length
  }
}
