from validators.SoftMaxModelInputValidator import SoftMaxModelInputValidator
from utils.ModelWeightInitializer import ModelWeightInitializer
from utils.Utility import Utility
import numpy as np

class SoftMax:

    def __init__(self, feature_data, label, learning_rate=0.01):
        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(-1, 1)
        labels = label.ravel().astype(int)
        self.input_validator= SoftMaxModelInputValidator(feature_data, label)
        self.feature_data=feature_data
        self.total_classes=np.unique(label).size
        self.label=np.eye(self.total_classes)[labels]
        self.learning_rate = learning_rate
        data_shape = feature_data.shape
        self.sample_count = data_shape[0]
        self.feature_count = data_shape[1]
        self._weight= ModelWeightInitializer.generateRandomWeightsVector(self.feature_count, self.total_classes)
        self._bias=ModelWeightInitializer.generateZeroWeightsVector(self.total_classes).ravel()
        self._loss=[]

    def train(self, iterations, logTrainingParams= False):
        self._loss=[]
        for i in range(iterations):
            
            probs = self.predict(self.feature_data)

            # Parital differentiation of cost function CE J = sum(-ylog(y_pred))/sample_count
            # def calculateCrossEntropyLoss(actual, predicted)
            # dJ/dz =  (probs - self.label) / self.sample_count
            # dJ/dW =  np.dot(self.feature_data.T , grad_z)
            # dJ/dB =  grad_z.sum(axis=0)

            grad_z = (probs - self.label) / self.sample_count

            dW = np.dot(self.feature_data.T , grad_z)
            dB = grad_z.sum(axis=0)

            self._weight = self._weight- self.learning_rate*dW
            self._bias = self._bias - self.learning_rate*dB
            loss = self.calculateCrossEntropyLoss(probs, self.label)
            self._loss.append(loss)

            if logTrainingParams and i%1000 == 0:
                print("Iteration ", i , " weights ", self.weight, " bias ", self.bias, "loss" , loss)

    def predict(self, attributes, predictOneHot=False):
        logits   = self.__getLogits(attributes)
        exp      = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabs  = exp / exp.sum(axis=1, keepdims=True)

        if predictOneHot:
            return probabs , Utility.createOneHotEncodingTop1(probabs)
        else:
            return probabs

    
    def __getLogits(self, attributes):
        return np.dot(attributes, self.weight) + self.bias
    
    def calculateCrossEntropyLoss(self, probs, labels):
        per_sample = np.sum(labels * np.log(probs + 1e-12), axis=1)
        return -np.mean(per_sample)

    @property
    def bias(self):
        return self._bias       

    @property
    def weight(self):
        return self._weight
    
    @property
    def trainingLoss(self):
        return self._loss