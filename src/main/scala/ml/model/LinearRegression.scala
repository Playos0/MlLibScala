package ml.model

import ml.data.Dataset
import ml.math.Vec
import ml.training.EarlyStopping


//Minimalny LinearRegression do przewidywania średniej etykiet w zbiorze danych

class LinearRegression(val learningRate: Double = 0.01,
                       val epochs: Int = 100,
                       val batchSize: Int = 16,
                       val lambda: Double,
                       val earlyStopping: Option[EarlyStopping] = None)
  extends Model {

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

        //nowy kod do gradientu z L2
        val gradient = batchX.zip(batchY).map { case (x, y) =>
          val xWithBias = x.withBias
          xWithBias * (weights.dot(xWithBias) - y)
        }.reduce(_ + _)

        //normalizuje tu gradient


        val dataGradient = gradient * (1.0 / batchX.size)
        //na koiniec aktualizacja wag

        //tu teraz konkretnie te L2
        val l2Gradient = Vec.fromVector(
          0.0 +: weights.tail.map(_ * lambda)
        )


        //final gradient
        val avgGradient = dataGradient + l2Gradient

        //i tu update wag
        weights = weights - (avgGradient * learningRate)
      }
      val predictions = data.features.map { x =>
        weights.dot(x.withBias)
      }

      val mse = Loss.mse(predictions,data.labels)
      println(f"Epoch $epoch%3d: MSE = $mse%.6f")

      earlyStopping.foreach { es =>
        if(es.shouldStop(mse)){
          println(s"Early Stopping at epoch $epoch")
          return new TrainedLinearRegression(weights)
        }
      }
    }

    println(s"Final weights: $weights")
    new TrainedLinearRegression(weights)
  }
}





