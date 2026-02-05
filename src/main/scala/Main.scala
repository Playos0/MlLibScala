import ml.data.Dataset
import ml.math
import ml.math.Vec
import ml.model.{LinearRegression, TrainedLinearRegression, TrainedModel}

object Main{
  def main(args: Array[String]): Unit ={


    //wejściowo dwie cechy
    val features = Vector(
      Vec(1.0, 2.0),
      Vec(2.0, 3.0)
    )

    val labels = Vector(5.0, 8.0)

    val dataset = Dataset(features, labels)

    val model = new LinearRegression()
    val trained = model.fit(dataset)

    val testInput = Vec(3.0,4.0)
    val prediction = trained.predict(testInput)

    println(s"Predykcja dla $testInput = $prediction")
  }
}