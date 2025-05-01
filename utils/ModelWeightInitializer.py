import numpy as np

class ModelWeightInitializer:
    
    @staticmethod
    def generateRandomWeightsVector(features , samples):
        weights = np.random.uniform(0, 0.9, size=(features, 1))
        return weights