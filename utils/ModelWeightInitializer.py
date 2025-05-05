import numpy as np

class ModelWeightInitializer:
    
    @staticmethod
    def generateRandomWeightsVector(features, rows=1):
        weights = np.random.uniform(0, 0.9, size=(features, rows))
        return weights

    @staticmethod
    def generateZeroWeightsVector(features, rows=1):
        weights = np.random.uniform(0, 0, size=(features, rows))
        return weights