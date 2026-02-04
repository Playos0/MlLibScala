package ml.data

case class Dataset[X,Y](features: Vector[X], labels: Vector[Y]){
  require(features.size == labels.size)
}
