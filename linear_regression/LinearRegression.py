import numpy as np
from validators.LinearRegressionInputValidator import LinearRegressionInputValidator
from utils.ModelWeightInitializer import ModelWeightInitializer

class LinearRegression:
    
    def __init__(self, feature_data, label, learning_rate=0.01):
        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(-1, 1)
        if label.ndim == 1:
            label = label.reshape(-1, 1)
        self.input_validator= LinearRegressionInputValidator(feature_data, label)
        self.feature_data=feature_data
        self.label=label
        self.learning_rate = learning_rate
        data_shape = feature_data.shape
        self.sample_count = data_shape[0]
        self.feature_count = 1 if len(data_shape) == 1 else data_shape[1]
        self._weight= ModelWeightInitializer.generateRandomWeightsVector(self.feature_count)
        self._bias=0
        self._loss=[]
    
    def train(self, iterations, logTrainingParams= False):

        self._loss=[]
        for i in range(iterations):
            predictions = np.dot(self.feature_data, self.weight) + self.bias

            # Parital differentiation of cost function MSE J = sum(( prediction - actual)^2)/sample_count
            # def calculateMseLoss(actual, predicted)
            # dJ/dW = 2*X*(prediction - actual)/sample_count
            # dJ/dB = 2*(prediction - actual)/sample_count
            error = predictions-self.label
            dW = ((2*np.dot(self.feature_data.T , error))/self.sample_count)
            dJ = ((2*np.sum(error))/self.sample_count)

            self._weight = self._weight- self.learning_rate*dW
            self._bias = self._bias - self.learning_rate*dJ
            loss = self.calculateMseLoss(self.label, predictions)
            self._loss.append(loss)

            if logTrainingParams and i%1000 == 0:
                print("Iteration ", i , " weights ", self.weight, " bias ", self.bias, "loss" , loss)

    def predict(self, attributes):
        return np.dot(attributes,self.weight)+self.bias
    
    def calculateMseLoss(self,actual, predicted):
        actual    = np.asarray(actual).reshape(-1)
        predicted = np.asarray(predicted).reshape(-1)
        if actual.shape != predicted.shape:
            raise ValueError("Length mismatch")
        return np.mean((actual - predicted) ** 2)
        

    @property
    def bias(self):
        return self._bias       

    @property
    def weight(self):
        return self._weight
    
    @property
    def trainingLoss(self):
        return self._loss