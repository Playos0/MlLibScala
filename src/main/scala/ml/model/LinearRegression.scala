package ml.model

import ml.data.Dataset
import ml.math.Vec


//Minimalny LinearRegression do przewidywania średniej etykiet w zbiorze danych

class LinearRegression(val learningRate: Double = 0.01, val epochs: Int = 100) extends Model {

  override def fit(data: Dataset): TrainedModel = {

    val numFeatures = data.features.head.size
    var weights = Vec.fill(numFeatures + 1)(0.0) //dodaje bias jako pierwszą wage
    val mseHistory = scala.collection.mutable.ArrayBuffer[Double]() //lista z historią zmian w MSE

    var bestMSE = Double.MaxValue
    var epochWithoutImprovement = 0

    val patience = 5
    val minDelta = 1e-6


    for(epoch <- 1 to epochs){
      //predykcja dla wszystkich danych (z bias)
      val predictions = data.features.map { x =>
        val xWithBias = x.withBias //Vec(1.0 +: x)
        weights.dot(xWithBias)
      }

      //najpierw trzeba obliczyć ten gradient
      val gradient = data.features.zip(data.labels).map { case (x, y) =>
        val xWithBias =x.withBias       //Vec(1.0 +: x)
        xWithBias * (weights.dot(xWithBias) - y) //tu mnoże wektor przez skalar (to co w nawiasie)
      }.reduce(_ + _) // sumje na koniec wszystkie wektroty w gradiecie

      // normalizacja przez liczbe przypadków i aktualizacja wag AI to jednak potrafi ładny komentarz napisać jak mu się kod wyśle XD
      weights = weights - (gradient * (learningRate / data.features.size))

      //to oblicza te MSE dla każdej iteracji (epoki) i wyświetla "log"
      val mse = Loss.mse(predictions, data.labels)
      mseHistory.append(mse)
      println(f"Epoch $epoch%3d: MSE = $mse%.6f")

      if (bestMSE - mse > minDelta){
        bestMSE = mse
        epochWithoutImprovement = 0
      }else{
        epochWithoutImprovement += 1
      }

      if(epochWithoutImprovement >= patience){
        println(s"Early stopping at epoch $epoch")
        println(s"Best MSE = $bestMSE")
        println(s"Final weights: $weights")
        return new TrainedLinearRegression(weights)
      }
    }

    println(s"Final weights: $weights")
    new TrainedLinearRegression(weights)
  }
}





