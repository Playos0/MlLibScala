package ml.model

import ml.data.Dataset
import ml.math.Vec


//Minimalny LinearRegression do przewidywania średniej etykiet w zbiorze danych

class LinearRegression(val learningRate: Double = 0.01, val epochs: Int = 100, val batchSize: Int = 16) extends Model {

  override def fit(data: Dataset): TrainedModel = {

    val numFeatures = data.features.head.size
    var weights = Vec.fill(numFeatures + 1)(0.0)  //dodaje bias jako pierwszą wage

    for(epoch <- 1 to epochs){

      val shuffled = scala.util.Random.shuffle(data.features.zip(data.labels)) //zipuje dane i shuffluje

      //dziele na mini "strefy" by uwydajnić model
      val batches = shuffled.grouped(batchSize)

      //pętla do nauki na każdą mini strefe
      for( batch <- batches){

        val (batchX, batchY) = batch.unzip

        val gradient = batchX.zip(batchY).map { case (x, y) => val xWithBias = x.withBias
          xWithBias * (weights.dot(xWithBias) - y)}.reduce(_ + _)

        //normalizuje tu gradient
        val avgGradient = gradient * (1.0 / batchX.size)

        //na koiniec aktualizacja wag
        weights = weights - (avgGradient *  learningRate)
      }
      val predictions = data.features.map { x =>
        weights.dot(x.withBias)
      }

      val mse = Loss.mse(predictions,data.labels)
      println(f"Epoch $epoch%3d: MSE = $mse%.6f")
    }

    println(s"Final weights: $weights")
    new TrainedLinearRegression(weights)
  }
}





