import ml.data.Dataset
import ml.math
import ml.math.Vec
import ml.model.LinearRegression

object Main{
  def main(args: Array[String]): Unit ={

    val features = Vector(
      Vec(1.0, 2.0),
      Vec(2.0, 3.0),
      Vec(3.0, 4.0)
    )

    val labels = Vector(10.0, 20.0, 30.0)

    val dataset = Dataset(features, labels)

    val model = new LinearRegression()
    val trainded = model.fit(dataset)

    val x = Vec(5.0, 6.0)
    println(trainded.predict(x))
  }
}