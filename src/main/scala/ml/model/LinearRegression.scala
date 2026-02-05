package ml.model

import ml.data.Dataset
import ml.math.Vec

//Minimalny LinearRegression do przewidywania średniej etykiet w zbiorze danych

class LinearRegression extends Model {

  override def fit(data: Dataset): TrainedModel = {

    //Liczba cech (z założeniem że wszystkie Vec są tego samego rozmiaru)
    val numFeatures = data.features.head.size

    //WAGI NA TEST SAME 1
    val weights = Vec.fill(numFeatures)(1.0)

    new TrainedLinearRegression(weights)
  }
}





