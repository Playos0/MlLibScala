import ml.data.Dataset
import ml.math
import ml.math.Vec
import ml.model.{LinearRegression, TrainedLinearRegression, TrainedModel, Loss}
import ml.preprocess.StandardScaler

object Main{
  def main(args: Array[String]): Unit ={


    /** wejściowo dwie cechy
    val features = Vector(
      Vec(1.0, 2.0),
      Vec(2.0, 3.0)
    )

    val labels = Vector(5.0, 8.0)

    val dataset = Dataset(features, labels)
     */

    val dataset = Dataset(
      Vector(Vec(1.0,2.0), Vec(2.0, 3.0)),
      Vector(5.0, 8.0)
    )

    val scaler = StandardScaler.fit(dataset.features)
    val scaledData = dataset.copy(features = scaler.transformAll(dataset.features))

    val model = new LinearRegression(learningRate = 0.1, epochs = 20)
    val trained = model.fit(scaledData)

    val predictions = scaledData.features.map { x =>
      trained.predict(x.withBias) //dodaje bias do predykcji
    }

    val mseValue = Loss.mse(predictions, scaledData.labels)

    println(s"Predykcja = $predictions")
    println(s"Mean Squared Error = $mseValue")
  }
}