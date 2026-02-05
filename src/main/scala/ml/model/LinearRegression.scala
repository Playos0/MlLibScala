package ml.model

import ml.data.Dataset
import ml.math.Vec

//Minimalny LinearRegression do przewidywania średniej etykiet w zbiorze danych

class LinearRegression(val learningRate: Double = 0.01, val epochs: Int = 100) extends Model {

  override def fit(data: Dataset): TrainedModel = {

    val numFeatures = data.features.head.size
    var weights = Vec.fill(numFeatures)(0.0) // zaczynam wszystko od zera

    for(_ <- 1 to epochs){
      //najpierw trzeba obliczyć ten gradient
      val gradient = data.features.zip(data.labels).map { case (x, y) =>
        val yHat = weights.dot(x)
        x * (yHat - y) //tu mnoże wektor przez skalar (to co w nawiasie)
      }.reduce(_ + _) // sumje na koniec wszystkie wektroty w gradiecie

      // normalizacja przez liczbe przypadków i aktualizacja wag AI to jednak potrafi ładny komentarz napisać jak mu się kod wyśle XD
      weights = weights - (gradient * (learningRate / data.features.size))
    }

    new TrainedLinearRegression(weights)
  }
}





