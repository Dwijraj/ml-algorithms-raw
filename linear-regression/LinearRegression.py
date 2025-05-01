import numpy as np
from validators.LinearRegressionInputValidator import LinearRegressionInputValidator

class LinearRegression:
    
    def __init__(self, feature_data, label):
        self.input_validator= LinearRegressionInputValidator(feature_data, label)
        self.feature_data=feature_data
        self.label=label
    
    def __train():
        pass


features = np.array([[1, 2], [3, 4]])
labels = np.array([10, 20])
l = LinearRegression(features, labels)