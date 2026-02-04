package ml.math


opaque type Vec = Vector[Double]

object Vec{

  //konstruktor do Vec

  def apply(values: Double*): Vec =
    values.toVector


  //metody do typu Vec (rozszerzenie)
  extension(v: Vec)

    def size: Int =
      v.length


    def +(other: Vec): Vec = //definicja metody dodawania dla obiektu Vec
      require(v.length == other.length) //sprawdza by dwa wektory były równej długości
      v.zip(other).map(_ + _)

    def *(scalar: Double): Vec =
      v.map(_ * scalar)

    def dot(other: Vec): Double =
      require(v.length == other.length)
      v.zip(other).map(_ * _).sum

    def norm: Double =
      math.sqrt(v.dot(v))
}



