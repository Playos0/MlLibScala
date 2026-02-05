package ml.math


opaque type Vec = Vector[Double]

object Vec{

  def fromVector(v: Vector[Double]): Vec =
    v

  //konstruktor do Vec

  def apply(values: Double*): Vec =
    values.toVector

  def fill(size: Int)(value: Double): Vec =
    Vector.fill(size)(value)


  //metody do typu Vec (rozszerzenie)
  extension(v: Vec)

    def size: Int =
      v.length


    def +(other: Vec): Vec = //definicja metody dodawania dla obiektu Vec
      require(v.length == other.length) //sprawdza by dwa wektory były równej długości
      v.zip(other).map(_ + _)

    def -(other: Vec): Vec =
      require(v.length == other.length)
      v.zip(other).map(_ - _)


    def *(scalar: Double): Vec =
      v.map(_ * scalar)

    def dot(other: Vec): Double =
      require(v.length == other.length)
      v.zip(other).map(_ * _).sum

    def norm: Double =
      math.sqrt(v.dot(v))


    //nowa metoda do bias
    def withBias: Vec =
      Vec.fromVector(1.0 +: v)
}



