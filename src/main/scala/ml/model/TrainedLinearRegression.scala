package ml.model

import ml.math.Vec

class TrainedLinearRegression(weight: Vec) extends TrainedModel {

  override def predict(features: Vec): Double = { //"nadpisanie" funkcji predict by pasowała
    //ŷ = w · x wiem że ten wzór ale w sumie nie wiem po co dla czego ale buja
    weight.dot(features)
  }
}