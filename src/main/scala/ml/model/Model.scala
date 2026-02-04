package ml.model

import ml.data.Dataset

/** jest coś co pozwala udostępniać interfejsy innym klasą */
trait Model  [X, Y]{
    def fit(data: Dataset[X, Y]): TrainedModel[X, Y]
}

trait TrainedModel[X, Y]{
    def predict(x: X): Y
}


