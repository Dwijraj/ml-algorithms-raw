import numpy as np
from validators.LinearRegressionInputValidator import LinearRegressionInputValidator
from utils.ModelWeightInitializer import ModelWeightInitializer

class LinearRegression:
    
    def __init__(self, feature_data, label, learning_rate=0.01):
        self.input_validator= LinearRegressionInputValidator(feature_data, label)
        self.feature_data=feature_data
        self.label=label
        self.learning_rate = learning_rate
        sample_count , feature_count = feature_data.shape
        self.sample_count = sample_count
        self.feature_count = feature_count
        self.weight= ModelWeightInitializer.generateRandomWeightsVector(self.feature_count)
        self.bias=0
    
    def train(self, iterations):

        for i in range(iterations):
            predictions = np.dot(self.feature_data, self.weight) + self.bias

            # Parital differentiation of cost function MSE J = sum(( prediction - actual)^2)/sample_count
            # dJ/dW = 2X(prediction - actual)/sample_count
            # dJ/dB = 2(prediction - actual)/sample_count

            dW = ((2*np.dot(self.feature_data , predictions-self.label))/self.sample_count)
            dJ = ((2*np.sum(predictions-self.label))/self.sample_count)

            self.weight = self.weight- self.learning_rate*dW
            self.bias = self.bias - self.learning_rate*dJ

            print("Iteration ", i , " weights ", self.weight, " bias ", self.bias)

    def predict(self, attributes):
        return np.dot(attributes,self.weight)+self.bias
        

    @property
    def bias(self):
        return self.bias       

    @property
    def weight(self):
        return self.weight

features = np.array([[1, 2], [3, 4]])
labels = np.array([10, 20])
l = LinearRegression(features, labels)