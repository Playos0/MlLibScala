package ml.model

import ml.data.Dataset
import ml.math.Vec

//Minimalny LinearRegression do przewidywania średniej etykiet w zbiorze danych

class LinearRegression extends Model {

  override def fit(data: Dataset): TrainedModel = {
    //oblicza średnią z etykiet
    val meanLabel = if(data.size > 0) data.labels.sum / data.labels.size else 0.0

    //Zwraca wytrenowany "model" XD
    new TrainedLinearRegression(meanLabel)
  }
}

class TrainedLinearRegression(mean: Double) extends TrainedModel { //klasa powiązana bezpośrednio z interfejsem Trained Model

  override def predict(features: Vec): Double = { //przyjmiuje wektor cech
    //na ten moment przewiduje tylko średnią wartość
    mean
  }
}