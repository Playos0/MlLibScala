package ml.training

class EarlyStopping(val patience: Int = 5 //"cierpliwość" do zatrzymywania
  ,val minDelta: Double = 1e-4 //ignorowanie szumu danych
){

  private var bestloss: Double = Double.PositiveInfinity
  private var epochWithoutImprovement = 0

  def shouldStop(currentLoss: Double): Boolean = {
    if(bestloss - currentLoss > minDelta) {
      bestloss = currentLoss
      epochWithoutImprovement = 0
      false
    } else {
      epochWithoutImprovement += 1
      epochWithoutImprovement >= patience
    }
  }
}


