package ml.model

import ml.data.Dataset
import ml.math.Vec

import java.awt.Taskbar.Feature

/** jest coś co pozwala udostępniać interfejsy innym klasą */

//interfejs dla każdego modelu
trait Model {
    def fit(data: Dataset): TrainedModel
}

//to po wytrenowaniu modelu
trait TrainedModel{
    def predict(features: Vec): Double
}

