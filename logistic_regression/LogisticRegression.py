from validators.LogisticRegressionInputValidator import LogisticRegressionInputValidator
from utils.ModelWeightInitializer import ModelWeightInitializer
import numpy as np

class LogisticRegression:

    def __init__(self, feature_data, label, learning_rate=0.01, threshold=0.5):
        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(-1, 1)
        if label.ndim == 1:
            label = label.reshape(-1, 1)
        self.input_validator= LogisticRegressionInputValidator(feature_data, label, threshold)
        self.feature_data=feature_data
        self.label=label
        self.learning_rate = learning_rate
        self.threshold= threshold
        data_shape = feature_data.shape
        self.sample_count = data_shape[0]
        self.feature_count = 1 if len(data_shape) == 1 else data_shape[1]
        self._weight= ModelWeightInitializer.generateRandomWeightsVector(self.feature_count)
        self._bias=0
        self._loss=[]

    def train(self, iterations, logTrainingParams= False):
        pass

    def predict(self, attributes):
        return 1 if self.__sigmoidFunction(np.dot(attributes,self.weight)+self.bias) >= self.threshold else 0
    
    def __sigmoidFunction(self, x):
            z = np.asarray(z, dtype=np.float64)          # accepts scalars & arrays
            # clip to avoid overflow when z is very negative/positive
            z = np.clip(z, -500, 500)                   
            return 1.0 / (1.0 + np.exp(-z))
    
    def calculateBinaryCrossEntropyLoss(self,actual, predicted):
        pass
        

    @property
    def bias(self):
        return self._bias       

    @property
    def weight(self):
        return self._weight
    
    @property
    def trainingLoss(self):
        return self._loss