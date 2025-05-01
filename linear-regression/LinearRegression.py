import numpy as np
from validators.LinearRegressionInputValidator import LinearRegressionInputValidator
from utils.ModelWeightInitializer import ModelWeightInitializer

class LinearRegression:
    
    def __init__(self, feature_data, label):
        self.input_validator= LinearRegressionInputValidator(feature_data, label)
        self.feature_data=feature_data
        self.label=label
        shape = feature_data.shape
        self.weight , self.bias = ModelWeightInitializer.generateRandomWeightsVector(shape[1], shape[0])
    
    def __train():
        pass

features = np.array([[1, 2], [3, 4]])
labels = np.array([10, 20])
l = LinearRegression(features, labels)