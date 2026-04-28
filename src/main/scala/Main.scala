import ml.data.Dataset
import ml.math
import ml.math.Vec
import ml.model.{LinearRegression, TrainedLinearRegression, TrainedModel, Loss}
import ml.preprocess.StandardScaler
import ml.metrics.Metrics
import ml.training.EarlyStopping


object Main{

  //tylko do testów
  def trainTestSplit[T](
                         features: Vector[T],
                         labels: Vector[Double],
                         testRatio: Double = 0.2
                       ): (Vector[T], Vector[Double], Vector[T], Vector[Double]) = {

    val n = features.size
    val split = (n * (1 - testRatio)).toInt
    val zipped = features.zip(labels)
    val shuffled = scala.util.Random.shuffle(zipped)
    val (train, test) = shuffled.splitAt(split)
    (train.map(_._1), train.map(_._2), test.map(_._1), test.map(_._2))
  }
  //tylko do testów

  def main(args: Array[String]): Unit ={


    /** wejściowo dwie cechy
    val features = Vector(
      Vec(1.0, 2.0),
      Vec(2.0, 3.0)
    )

    val labels = Vector(5.0, 8.0)

    val dataset = Dataset(features, labels)
     */

    //generowanie danych do testowania
    //20 punktów + szum
    val features = (1 to 100).map(i => Vec(i.toDouble, (i * 2 ).toDouble)).toVector
    val labels = features.map(v => 2 * v(0) + 3 * v(1) + 5 + scala.util.Random.nextGaussian() * 0.5)

    val dataset = Dataset(features, labels)

    val (trainX,  trainY, testX, testY) = trainTestSplit(dataset.features, dataset.labels, 0.25)//do testów

    val scaler = StandardScaler.fit(dataset.features)
    val scaledData = dataset.copy(features = scaler.transformAll(dataset.features))
    val trainXScaled = trainX.map(scaler.transform)
    val testXScaled = testX.map(scaler.transform)

    val model = new LinearRegression(learningRate = 0.1,
      epochs = 20,
      batchSize = 16,
      lambda = 0.1,
      earlyStopping = Some(new EarlyStopping(patience = 5)))

    val trained = model.fit(Dataset(trainXScaled, trainY))
    //val trained = model.fit(scaledData)

    //val predictions = scaledData.features.map { x =>
    //  trained.predict(x.withBias) //dodaje bias do predykcji
    //}
    val trainPred = trainXScaled.map(x => trained.predict(x.withBias))
    val testPred = testXScaled.map(x => trained.predict(x.withBias))


    //val mseValue = Loss.mse(predictions, scaledData.labels)

    val trainMSE = Loss.mse(trainPred, trainY)
    val testMSE = Loss.mse(testPred, testY)

    //println(s"Predykcja = $predictions")
    //println(s"Mean Squared Error = $mseValue")
    //println(s"Train R² = ${Metrics.r2(trainPred, trainY)}")
    //println(s"Test  R² = ${Metrics.r2(testPred, testY)}")

    println(s"Train Predykcja = $trainPred")
    println(s"Test  Predykcja = $testPred")
    println(s"Train MSE = $trainMSE")
    println(s"Test  MSE = $testMSE")
    println(s"Train R² = ${Metrics.r2(trainPred, trainY)}")
    println(s"Test  R² = ${Metrics.r2(testPred, testY)}")
  }
}