package ml.data

import ml.math.Vec


//ta klasa przechowuje cechy i etykiety danych
case class Dataset(features: Vector[Vec], //każda obserwacja to wektor cech
                   labels: Vector[Double]) { // a to odpowiadające im etykiety
  require(features.size == labels.size, "Liczba cech musi odpowiadać liczbie etykiet")

  def size: Int = features.size

  //dzieli na train/test (na ten moment łatwa wersja)
  def split(ratio: Double = 0.8): (Dataset, Dataset) = { //ratio to "scala" podziału
    val splitIndex = (features.size * ratio).toInt
    val train = Dataset(features.take(splitIndex), labels.take(splitIndex))
    val test = Dataset(features.drop(splitIndex), labels.drop(splitIndex))
    (train, test)
  }
}


